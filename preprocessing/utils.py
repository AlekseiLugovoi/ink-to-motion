import io
import cv2
import numpy as np
from pathlib import Path
from lxml import etree
from PIL import Image as PILImage

ARUCO_DICT = cv2.aruco.DICT_4X4_50


# ---------------------------------------------------------------------------
#  SVG → PNG
# ---------------------------------------------------------------------------

def render_svg(svg_path, remove_clip=True, scale=1, green_to_white=False):
    """Render SVG to RGBA numpy array.

    scale: render at Nx resolution (e.g. scale=3 → 3x bigger PNG).
    green_to_white: replace #00FF00 fills with white (for template overlay).
    """
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM

    tree = etree.parse(svg_path)
    root = tree.getroot()

    needs_tmp = remove_clip or green_to_white

    if remove_clip:
        for g in root.iter("{http://www.w3.org/2000/svg}g"):
            if g.get("clip-path"):
                del g.attrib["clip-path"]
        for defs in root.findall("{http://www.w3.org/2000/svg}defs"):
            defs.getparent().remove(defs)

    if green_to_white:
        for p in root.iter("{http://www.w3.org/2000/svg}path"):
            if (p.get("fill") or "").upper() == "#00FF00":
                p.set("fill", "white")

    if needs_tmp:
        tmp = str(Path(svg_path).with_suffix(".tmp.svg"))
        tree.write(tmp, xml_declaration=True)
        drawing = svg2rlg(tmp)
        Path(tmp).unlink()
    else:
        drawing = svg2rlg(svg_path)

    if scale != 1:
        drawing.width *= scale
        drawing.height *= scale
        drawing.scale(scale, scale)

    buf = io.BytesIO()
    renderPM.drawToFile(drawing, buf, fmt="PNG")
    return np.array(PILImage.open(buf).convert("RGBA"))


MARKER_IDS = [0, 1, 2, 3]  # TL, TR, BR, BL


# ---------------------------------------------------------------------------
#  Template geometry (pure math, no I/O)
# ---------------------------------------------------------------------------

def template_meta(canvas_size=(1240, 1754), marker_px=80, margin=60, content_pad=10):
    """Compute template geometry from parameters.

    Returns dict with marker_centers, drawing_bbox, canvas_size.
    Same structure used by detect_and_align / extract_colored.
    """
    canvas_h, canvas_w = canvas_size
    inner = margin + marker_px + content_pad
    drawing_bbox = [inner, inner, canvas_w - 2 * inner, canvas_h - 2 * inner]

    positions = [
        (margin, margin),
        (canvas_w - margin - marker_px, margin),
        (canvas_w - margin - marker_px, canvas_h - margin - marker_px),
        (margin, canvas_h - margin - marker_px),
    ]
    marker_centers = [
        [px + marker_px / 2, py + marker_px / 2] for px, py in positions
    ]

    return {
        "marker_centers": marker_centers,
        "drawing_bbox": drawing_bbox,
        "canvas_size": [canvas_h, canvas_w],
    }


# ---------------------------------------------------------------------------
#  Decorative helpers
# ---------------------------------------------------------------------------

def _draw_hatching(canvas, drawing_bbox, marker_positions, marker_px,
                    spacing=10, color=(200, 200, 200), thickness=1, corner_radius=20):
    """Draw diagonal hatching outside the drawing area and a rounded border."""
    h, w = canvas.shape[:2]
    bx, by, bw, bh = drawing_bbox

    # rounded rectangle mask for drawing area (white inside)
    mask = np.zeros((h, w), dtype=np.uint8)
    r = corner_radius
    cv2.rectangle(mask, (bx + r, by), (bx + bw - r, by + bh), 255, -1)
    cv2.rectangle(mask, (bx, by + r), (bx + bw, by + bh - r), 255, -1)
    cv2.circle(mask, (bx + r, by + r), r, 255, -1)
    cv2.circle(mask, (bx + bw - r, by + r), r, 255, -1)
    cv2.circle(mask, (bx + bw - r, by + bh - r), r, 255, -1)
    cv2.circle(mask, (bx + r, by + bh - r), r, 255, -1)

    # clear zones around markers
    pad = 4
    marker_mask = np.zeros((h, w), dtype=np.uint8)
    for (px, py) in marker_positions:
        x1 = max(0, px - pad)
        y1 = max(0, py - pad)
        x2 = min(w, px + marker_px + pad)
        y2 = min(h, py + marker_px + pad)
        cv2.rectangle(marker_mask, (x1, y1), (x2, y2), 255, -1)

    # diagonal lines, then mask
    line_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    diag = h + w
    for offset in range(-diag, diag, spacing):
        pt1 = (offset, 0)
        pt2 = (offset + h, h)
        cv2.line(line_canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    hatch_mask = (mask == 0) & (marker_mask == 0)
    canvas[hatch_mask] = line_canvas[hatch_mask]

    # rounded rectangle border
    border_color = (160, 160, 160)
    border_t = 2
    cv2.line(canvas, (bx + r, by), (bx + bw - r, by), border_color, border_t, cv2.LINE_AA)
    cv2.line(canvas, (bx + r, by + bh), (bx + bw - r, by + bh), border_color, border_t, cv2.LINE_AA)
    cv2.line(canvas, (bx, by + r), (bx, by + bh - r), border_color, border_t, cv2.LINE_AA)
    cv2.line(canvas, (bx + bw, by + r), (bx + bw, by + bh - r), border_color, border_t, cv2.LINE_AA)
    cv2.ellipse(canvas, (bx + r, by + r), (r, r), 180, 0, 90, border_color, border_t, cv2.LINE_AA)
    cv2.ellipse(canvas, (bx + bw - r, by + r), (r, r), 270, 0, 90, border_color, border_t, cv2.LINE_AA)
    cv2.ellipse(canvas, (bx + bw - r, by + bh - r), (r, r), 0, 0, 90, border_color, border_t, cv2.LINE_AA)
    cv2.ellipse(canvas, (bx + r, by + bh - r), (r, r), 90, 0, 90, border_color, border_t, cv2.LINE_AA)


# ---------------------------------------------------------------------------
#  1. Generate blank template
# ---------------------------------------------------------------------------

def generate_template(
    output_path="output/template.png",
    canvas_size=(1240, 1754),
    marker_px=80,
    margin=60,
    content_pad=10,
    decorate=True,
    hatch_spacing=10,
    hatch_color=(200, 200, 200),
    corner_radius=20,
    title="Sketch Factory",
):
    """Generate blank printable A4 template with ArUco markers and decoration.

    Saves a single PNG file. No JSON — use template_meta() to get geometry.
    """
    canvas_h, canvas_w = canvas_size
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    inner = margin + marker_px + content_pad
    drawing_bbox = [inner, inner, canvas_w - 2 * inner, canvas_h - 2 * inner]

    # ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    positions = [
        (margin, margin),
        (canvas_w - margin - marker_px, margin),
        (canvas_w - margin - marker_px, canvas_h - margin - marker_px),
        (margin, canvas_h - margin - marker_px),
    ]

    marker_border = 6
    marker_radius = 8
    marker_gray = 70
    for mid, (px, py) in zip(MARKER_IDS, positions):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_px)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        marker_bgr[marker_img == 0] = marker_gray

        # rounded corner mask
        rmask = np.zeros((marker_px, marker_px), dtype=np.uint8)
        mr = marker_radius
        cv2.rectangle(rmask, (mr, 0), (marker_px - mr, marker_px), 255, -1)
        cv2.rectangle(rmask, (0, mr), (marker_px, marker_px - mr), 255, -1)
        cv2.circle(rmask, (mr, mr), mr, 255, -1)
        cv2.circle(rmask, (marker_px - mr, mr), mr, 255, -1)
        cv2.circle(rmask, (marker_px - mr, marker_px - mr), mr, 255, -1)
        cv2.circle(rmask, (mr, marker_px - mr), mr, 255, -1)

        roi = canvas[py : py + marker_px, px : px + marker_px]
        roi[rmask > 0] = marker_bgr[rmask > 0]

        # outer frame with quiet zone
        b = marker_border
        x1, y1 = px - b, py - b
        x2, y2 = px + marker_px + b, py + marker_px + b
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), b, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (140, 140, 140), 2, cv2.LINE_AA)
        roi = canvas[py : py + marker_px, px : px + marker_px]
        roi[rmask > 0] = marker_bgr[rmask > 0]

    # decoration
    if decorate:
        _draw_hatching(canvas, drawing_bbox, positions, marker_px,
                       spacing=hatch_spacing, color=hatch_color,
                       corner_radius=corner_radius)

    # title
    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = margin + marker_px + 30
        text_y = margin + marker_px - 10
        cv2.putText(canvas, title, (text_x, text_y), font,
                    1.8, (120, 120, 120), 3, cv2.LINE_AA)

    # save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)

    return {"template": canvas, "template_path": str(out)}


# ---------------------------------------------------------------------------
#  2. Overlay character onto template
# ---------------------------------------------------------------------------

def image_rect_from_svg(svg_path, content_scale=1.0,
                        canvas_size=(1240, 1754), marker_px=80, margin=60, content_pad=10):
    """Вычислить image_rect из размеров SVG + геометрии шаблона.

    Возвращает [x0, y0, w, h] — координаты персонажа на шаблоне.
    Не требует PNG-файлов.
    """
    tree = etree.parse(str(svg_path))
    root = tree.getroot()
    svg_w = float(root.get("width"))
    svg_h = float(root.get("height"))

    meta = template_meta(canvas_size, marker_px, margin, content_pad)
    bx, by, bw, bh = meta["drawing_bbox"]

    scale = min(bw / svg_w, bh / svg_h) * content_scale
    tw, th = int(svg_w * scale), int(svg_h * scale)
    x0 = bx + (bw - tw) // 2
    y0 = by + (bh - th) // 2
    return [x0, y0, tw, th]


def overlay_character(template_path, svg_path, output_path=None, content_scale=1.0,
                      canvas_size=(1240, 1754), marker_px=80, margin=60, content_pad=10):
    """Наложить персонажа (из SVG) на шаблон.

    Рендерит SVG (зелёное→белое) и альфа-композитит на template.
    Возвращает composite image и image_rect.
    """
    template = cv2.imread(str(template_path))
    if template is None:
        raise FileNotFoundError(f"Cannot read template: {template_path}")

    img = cv2.cvtColor(
        render_svg(svg_path, green_to_white=True), cv2.COLOR_RGBA2BGRA
    )

    image_rect = image_rect_from_svg(
        svg_path, content_scale, canvas_size, marker_px, margin, content_pad
    )
    x0, y0, tw, th = image_rect
    resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    canvas = template.copy()
    alpha = resized[:, :, 3] / 255.0
    for c in range(3):
        canvas[y0 : y0 + th, x0 : x0 + tw, c] = (
            resized[:, :, c] * alpha
            + canvas[y0 : y0 + th, x0 : x0 + tw, c] * (1 - alpha)
        ).astype(np.uint8)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return {"template": canvas, "image_rect": image_rect, "output_path": output_path}


# ---------------------------------------------------------------------------
#  3. Alignment pipeline
# ---------------------------------------------------------------------------

def detect_and_align(img_path, meta=None):
    """Detect ArUco markers on photo, warp back to template coordinates.

    If meta is None, uses default template_meta().
    """
    if meta is None:
        meta = template_meta()

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = detector.detectMarkers(img)

    if ids is None or len(ids) < 4:
        found = 0 if ids is None else len(ids)
        raise ValueError(f"Expected 4 markers, found {found}")

    detected = {}
    for i, mid in enumerate(ids.flatten()):
        detected[int(mid)] = corners[i][0].mean(axis=0)

    src_pts = np.array(
        [detected[mid] for mid in MARKER_IDS], dtype=np.float32
    )
    dst_pts = np.array(meta["marker_centers"], dtype=np.float32)

    canvas_h, canvas_w = meta["canvas_size"]
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(
        img, M, (canvas_w, canvas_h), borderValue=(255, 255, 255)
    )

    return {"aligned": aligned, "matrix": M, "detected_centers": src_pts}


def alignment_error(aligned_img, meta=None, marker_px=80, margin=60):
    """Re-detect marker corners on aligned image, compare to expected.

    Returns mean corner error in pixels.
    Guideline: <1px excellent, 1-3px good, >3px consider retaking.
    """
    if meta is None:
        meta = template_meta()

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = detector.detectMarkers(aligned_img)

    if ids is None:
        return float("inf"), []

    canvas_h, canvas_w = meta["canvas_size"]
    positions = [
        (margin, margin),
        (canvas_w - margin - marker_px, margin),
        (canvas_w - margin - marker_px, canvas_h - margin - marker_px),
        (margin, canvas_h - margin - marker_px),
    ]

    expected_corners = {}
    for mid, (px, py) in zip(MARKER_IDS, positions):
        expected_corners[mid] = np.array([
            [px, py], [px + marker_px, py],
            [px + marker_px, py + marker_px], [px, py + marker_px],
        ], dtype=np.float32)

    errors = []
    for i, mid in enumerate(ids.flatten()):
        mid = int(mid)
        if mid in expected_corners:
            for d, e in zip(corners[i][0], expected_corners[mid]):
                errors.append(float(np.linalg.norm(d - e)))

    mean_err = np.mean(errors) if errors else float("inf")
    return mean_err, errors


def extract_colored(aligned_img, meta=None, white_thresh=230):
    """Crop drawing region from aligned image, white -> transparent."""
    if meta is None:
        meta = template_meta()

    x, y, w, h = meta["drawing_bbox"]
    crop = aligned_img[y : y + h, x : x + w]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray < white_thresh).astype(np.uint8) * 255

    b, g, r = cv2.split(crop)
    rgba = cv2.merge([b, g, r, mask])

    return {"rgba": rgba, "crop": crop, "mask": mask}


def correct_photo(aligned_img, image_rect, meta=None,
                   white_target=200, gain_range=(0.9, 1.2)):
    """White balance + brightness коррекция по белизне бумаги.

    Логика: бумага = поля вокруг персонажа (между drawing_bbox и image_rect).
    Это 100% белый листок. Усредняем его цвет на фото,
    вычисляем поканальный gain = white_target / paper_median.
    Например бумага [177, 181, 181] → gain [1.13, 1.10, 1.10].

    white_target: целевая яркость бумаги (200 = чуть теплее, 240 = ярче).
    gain_range: (min, max) ограничения на gain, защита от пересвета/недосвета.
    """
    if meta is None:
        meta = template_meta()

    ix, iy, iw, ih = image_rect
    bx, by, bw, bh = meta["drawing_bbox"]

    # Бумага = поля вокруг персонажа внутри drawing_bbox
    strips = []
    if iy > by:       strips.append(aligned_img[by:iy, bx:bx+bw])
    if iy+ih < by+bh: strips.append(aligned_img[iy+ih:by+bh, bx:bx+bw])
    if ix > bx:       strips.append(aligned_img[by:by+bh, bx:ix])
    if ix+iw < bx+bw: strips.append(aligned_img[by:by+bh, ix+iw:bx+bw])

    paper_pixels = np.vstack([s.reshape(-1, 3) for s in strips if s.size > 0])
    # Медиана устойчивее к выбросам (штриховка, тени маркеров)
    paper_ref = np.median(paper_pixels, axis=0)
    gain = np.clip(white_target / (paper_ref + 1e-6), *gain_range)

    corrected = (aligned_img.astype(np.float32) * gain[np.newaxis, np.newaxis, :])
    result = np.clip(corrected, 0, 255).astype(np.uint8)

    info = {"paper_bgr": paper_ref, "gain_bgr": gain}
    return result, info


def _svg_parse_paths(svg_path, sx, sy, pts_per_seg=30):
    """Парсим SVG path'ы → упорядоченный список операций рисования.

    Каждая операция: (subpath_pts, fill_color, stroke_color, stroke_width)
    fill_color/stroke_color = BGR tuple или None.
    Порядок = порядок в SVG (painter's algorithm).
    """
    from svgpathtools import parse_path

    tree = etree.parse(str(svg_path))
    root = tree.getroot()

    COLOR_MAP = {
        "#00ff00": (0, 255, 0), "#00FF00": (0, 255, 0),
        "black": (0, 0, 0), "#000000": (0, 0, 0), "#000": (0, 0, 0),
        "white": (255, 255, 255), "#ffffff": (255, 255, 255), "#FFFFFF": (255, 255, 255),
    }

    ops = []
    for path_el in root.iter("{http://www.w3.org/2000/svg}path"):
        d = path_el.get("d", "").strip()
        if not d:
            continue

        fill_str = (path_el.get("fill") or "none").strip()
        stroke_str = (path_el.get("stroke") or "none").strip()
        stroke_w = float(path_el.get("stroke-width") or "1")

        fill_bgr = COLOR_MAP.get(fill_str)
        stroke_bgr = COLOR_MAP.get(stroke_str)

        path_obj = parse_path(d)

        # Разбиваем на субпути (по разрывам)
        subpaths = []
        current = []
        for seg in path_obj:
            start = (seg.start.real * sx, seg.start.imag * sy)
            if current:
                last = current[-1]
                if ((start[0] - last[0])**2 + (start[1] - last[1])**2) > 4:
                    if len(current) >= 3:
                        subpaths.append(np.array(current, dtype=np.int32))
                    current = []
            if not current:
                current.append(start)
            for i in range(1, pts_per_seg + 1):
                pt = seg.point(i / pts_per_seg)
                current.append((pt.real * sx, pt.imag * sy))
        if len(current) >= 3:
            subpaths.append(np.array(current, dtype=np.int32))

        thickness = 1
        for pts in subpaths:
            ops.append((pts, fill_bgr, stroke_bgr, thickness))

    return ops


def transfer_color(aligned_img, svg_path, image_rect):
    """Перенос цвета с фото на оригинал — напрямую из SVG.

    SVG path'ы → полигоны → cv2.fillPoly/polylines.
    Никакой растеризации PNG — идеальные границы при любом разрешении.

    Painter's algorithm (порядок SVG):
    1. Зелёные fill → зона заливки цветом с фото
    2. Белые/чёрные fill → рисуем поверх (глаз, зрачок и пр.)
    3. Stroke → контуры поверх всего
    """
    ix, iy, iw, ih = image_rect
    photo_crop = aligned_img[iy : iy + ih, ix : ix + iw]

    tree = etree.parse(str(svg_path))
    root = tree.getroot()
    svg_w = float(root.get("width"))
    svg_h = float(root.get("height"))
    sx, sy = iw / svg_w, ih / svg_h

    ops = _svg_parse_paths(svg_path, sx, sy)

    # 1. Маска заливки (зелёное) — полигоны, идеальные границы
    green_bgr = (0, 255, 0)
    fill_mask = np.zeros((ih, iw), dtype=np.uint8)
    for pts, fill_bgr, _, _ in ops:
        if fill_bgr == green_bgr:
            cv2.fillPoly(fill_mask, [pts], 255)

    kernel = np.ones((7, 7), np.uint8)
    fill_expanded = cv2.dilate(fill_mask, kernel, iterations=1)

    # 2. Сборка в порядке SVG (painter's algorithm)
    result = np.full((ih, iw, 3), 255, dtype=np.uint8)
    result[fill_expanded > 0] = photo_crop[fill_expanded > 0]

    # Альфа по расширенной зоне
    final_alpha = np.zeros((ih, iw), dtype=np.uint8)
    final_alpha[fill_expanded > 0] = 255

    # Не-зелёные fills поверх (глаз, зрачок)
    for pts, fill_bgr, _, _ in ops:
        if fill_bgr and fill_bgr != green_bgr:
            cv2.fillPoly(result, [pts], fill_bgr)
            cv2.fillPoly(final_alpha, [pts], 255)

    # Контуры поверх всего
    for pts, _, stroke_bgr, thickness in ops:
        if stroke_bgr:
            cv2.polylines(result, [pts], False, stroke_bgr, thickness, cv2.LINE_AA)
            cv2.polylines(final_alpha, [pts], False, 255, thickness, cv2.LINE_AA)

    rgba = cv2.merge([*cv2.split(result), final_alpha])
    return {"rgba": rgba, "result": result, "is_fill": fill_mask > 0}


# ---------------------------------------------------------------------------
#  5. Skeleton markup
# ---------------------------------------------------------------------------

def _load_display_base(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    ih, iw = img.shape[:2]
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4] / 255.0
        bgr = img[:, :, :3]
        display = (bgr * alpha + 255 * (1 - alpha)).astype(np.uint8)
    else:
        display = img.copy()
    return display, iw, ih


def markup_skeleton(img_path, point_names=None):
    """Интерактивная разметка скелета.

    Два режима:
    - point_names задан (список) → кликаешь точки по порядку
    - point_names=None → LMB ставит точку, набираешь имя с клавиатуры, Enter подтверждает

    RMB = undo, ESC = готово. Backspace = стереть символ.
    Возвращает keypoints: {"name": [nx, ny], ...} — нормализованные 0..1.
    """
    display_base, iw, ih = _load_display_base(img_path)
    points = []         # [(x, y, name), ...]
    pending = [None]    # (x, y) ждёт имя
    typing = [""]       # текущий набираемый текст

    def redraw():
        display = display_base.copy()
        for x, y, name in points:
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(display, (x, y), 6, (255, 255, 255), 2)
            cv2.putText(display, name, (x + 10, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)
        if pending[0]:
            px, py = pending[0]
            cv2.circle(display, (px, py), 8, (0, 150, 255), 2)
            label = typing[0] + "_" if typing[0] else "type name..."
            cv2.putText(display, label, (px + 10, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 1)
        n = len(points)
        if point_names:
            total = len(point_names)
            if n < total:
                txt = f"Click: {point_names[n]} ({n}/{total}) | RMB=undo"
            else:
                txt = f"Done ({total}/{total})! Press any key"
        elif pending[0]:
            txt = f"[{n} pts] Type name, Enter=ok, Backspace=del"
        else:
            txt = f"[{n} pts] LMB=add, RMB=undo, ESC=done"
        cv2.putText(display, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
        cv2.imshow("Skeleton Markup", display)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if point_names:
                if len(points) < len(point_names):
                    points.append((x, y, point_names[len(points)]))
                    redraw()
            elif not pending[0]:
                pending[0] = (x, y)
                typing[0] = ""
                redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if pending[0]:
                pending[0] = None
                typing[0] = ""
            elif points:
                points.pop()
            redraw()

    redraw()
    cv2.setMouseCallback("Skeleton Markup", on_mouse)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        if point_names:
            if len(points) >= len(point_names):
                break
            continue
        if pending[0]:
            if key == 13:  # Enter
                if typing[0]:
                    points.append((*pending[0], typing[0]))
                pending[0] = None
                typing[0] = ""
                redraw()
            elif key == 8:  # Backspace
                typing[0] = typing[0][:-1]
                redraw()
            elif 32 <= key < 127:  # printable ASCII
                typing[0] += chr(key)
                redraw()

    cv2.destroyAllWindows()

    keypoints = {}
    for x, y, name in points:
        keypoints[name] = [round(x / iw, 4), round(y / ih, 4)]
    return keypoints


def draw_skeleton(img_path, keypoints):
    """Нарисовать точки скелета поверх изображения. Возвращает BGR numpy array."""
    display, iw, ih = _load_display_base(img_path)
    for name, (nx, ny) in keypoints.items():
        x, y = int(nx * iw), int(ny * ih)
        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.circle(display, (x, y), 6, (255, 255, 255), 2)
        cv2.putText(display, name, (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)
    return display


def save_skeleton(img_path, keypoints, output_dir, char_id=None):
    """Сохранить skeleton.json и skeleton.png.

    output_dir: папка для сохранения (например templates/001/).
    Возвращает пути к сохранённым файлам.
    """
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    ih, iw = img.shape[:2]

    data = {
        "image": str(img_path),
        "image_size": [iw, ih],
        "keypoints": keypoints,
    }
    if char_id is not None:
        data["char_id"] = char_id

    json_path = out / "skeleton.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    vis = draw_skeleton(img_path, keypoints)
    png_path = out / "skeleton.png"
    cv2.imwrite(str(png_path), vis)

    return {"json_path": str(json_path), "png_path": str(png_path)}


# ---------------------------------------------------------------------------
#  6. Animation engine
# ---------------------------------------------------------------------------

def warp_triangle(src_img, src_tri, dst_tri, dst_img):
    r = cv2.boundingRect(np.float32([dst_tri]))
    x, y, w, h = r
    x2 = min(x + w, dst_img.shape[1])
    y2 = min(y + h, dst_img.shape[0])
    x, y = max(x, 0), max(y, 0)
    w, h = x2 - x, y2 - y
    if w <= 0 or h <= 0:
        return
    dst_local = np.float32([(p[0] - x, p[1] - y) for p in dst_tri])
    M = cv2.getAffineTransform(np.float32(src_tri), dst_local)
    warped = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_local), 255)
    roi = dst_img[y:y2, x:x2]
    m = mask[:roi.shape[0], :roi.shape[1]] > 0
    warped = warped[:roi.shape[0], :roi.shape[1]]
    for c in range(dst_img.shape[2]):
        roi[:, :, c] = np.where(m, warped[:, :, c], roi[:, :, c])


def make_motion_from_config(t, base_pts, n_kp, names, motion_cfg):
    pts = base_pts.copy()
    a = t * 2 * np.pi
    for i, name in enumerate(names[:n_kp]):
        if name in motion_cfg:
            m = motion_cfg[name]
            angle = a * m["freq"] + m.get("phase", 0) * 2 * np.pi
            pts[i, 0] += m["dx"] * np.sin(angle)
            pts[i, 1] += m["dy"] * np.sin(angle)
    return pts


def draw_triangulation(img, ctrl_pts, tri, n_kp, names):
    """Draw triangulation mesh + keypoint labels over image. Returns RGBA."""
    vis = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) if img.shape[2] == 4 else img.copy()
    for simplex in tri.simplices:
        pts = ctrl_pts[simplex].astype(np.int32)
        cv2.polylines(vis, [pts], True, (200, 100, 0, 255), 1)
    for pt, name in zip(ctrl_pts[:n_kp], names):
        x, y = pt.astype(int)
        cv2.circle(vis, (x, y), 5, (0, 0, 255, 255), -1)
        cv2.putText(vis, name, (x + 8, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 0, 255), 1)
    return vis


def build_triangulation(keypoints, tw, th):
    from scipy.spatial import Delaunay
    names = list(keypoints.keys())
    skel_pts = np.array(
        [[kp[0] * tw, kp[1] * th] for kp in keypoints.values()],
        dtype=np.float32,
    )
    corners = np.array([
        [0, 0], [tw, 0], [tw, th], [0, th],
        [tw // 2, 0], [tw, th // 2], [tw // 2, th], [0, th // 2],
    ], dtype=np.float32)
    ctrl_pts = np.vstack([skel_pts, corners])
    tri = Delaunay(ctrl_pts)
    return ctrl_pts, tri, len(skel_pts), names


def animate_character(img_rgba, keypoints, motion_cfg, fps=15, duration=2):
    """Генерирует список BGRA-кадров анимации."""
    th, tw = img_rgba.shape[:2]
    if img_rgba.shape[2] == 3:
        img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2BGRA)
    ctrl_pts, tri, n_kp, names = build_triangulation(keypoints, tw, th)
    n_frames = fps * duration
    frames = []
    for i in range(n_frames):
        t = i / n_frames
        dst_pts = make_motion_from_config(t, ctrl_pts, n_kp, names, motion_cfg)
        dst_img = img_rgba.copy()
        for simplex in tri.simplices:
            warp_triangle(img_rgba, ctrl_pts[simplex].tolist(),
                          dst_pts[simplex].tolist(), dst_img)
        frames.append(dst_img)
    return frames
