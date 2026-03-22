import cv2
import numpy as np
from pathlib import Path

ARUCO_DICT = cv2.aruco.DICT_4X4_50
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

def overlay_character(template_path, img_path, output_path=None, content_scale=1.0,
                      canvas_size=(1240, 1754), marker_px=80, margin=60, content_pad=10):
    """Place a character image onto a template inside the drawing area.

    Returns the composite image and image_rect (position/size of the character).
    """
    template = cv2.imread(str(template_path))
    if template is None:
        raise FileNotFoundError(f"Cannot read template: {template_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    meta = template_meta(canvas_size, marker_px, margin, content_pad)
    bx, by, bw, bh = meta["drawing_bbox"]

    ih, iw = img.shape[:2]
    scale = min(bw / iw, bh / ih) * content_scale
    tw, th = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    x0 = bx + (bw - tw) // 2
    y0 = by + (bh - th) // 2
    image_rect = [x0, y0, tw, th]

    canvas = template.copy()
    if resized.shape[2] == 4:
        alpha = resized[:, :, 3] / 255.0
        for c in range(3):
            canvas[y0 : y0 + th, x0 : x0 + tw, c] = (
                resized[:, :, c] * alpha
                + canvas[y0 : y0 + th, x0 : x0 + tw, c] * (1 - alpha)
            ).astype(np.uint8)
    else:
        canvas[y0 : y0 + th, x0 : x0 + tw] = resized

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


def transfer_color(aligned_img, original_path, image_rect, line_thresh=100, sat_thresh=15):
    """Transfer colors from aligned photo onto the original digital drawing.

    image_rect: [x, y, w, h] — position of the character on the template
    (returned by overlay_character).
    """
    ix, iy, iw, ih = image_rect

    photo_crop = aligned_img[iy : iy + ih, ix : ix + iw]

    original = cv2.imread(str(original_path), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise FileNotFoundError(f"Cannot read: {original_path}")
    original = cv2.resize(original, (iw, ih), interpolation=cv2.INTER_AREA)

    if original.shape[2] == 4:
        alpha_ch = original[:, :, 3:4] / 255.0
        original_bgr = (original[:, :, :3] * alpha_ch + 255 * (1 - alpha_ch)).astype(np.uint8)
        orig_alpha = original[:, :, 3]
    else:
        original_bgr = original
        orig_alpha = None

    result = original_bgr.copy()

    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    is_line = (original_gray < line_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    is_line_dilated = cv2.dilate(is_line, kernel).astype(bool)
    is_fill = ~is_line_dilated

    photo_hsv = cv2.cvtColor(photo_crop, cv2.COLOR_BGR2HSV)
    is_colored = photo_hsv[:, :, 1] > sat_thresh

    mask = is_fill & is_colored
    result[mask] = photo_crop[mask]

    if orig_alpha is not None:
        alpha = orig_alpha.copy()
        alpha[mask] = 255
    else:
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        alpha = (result_gray < 230).astype(np.uint8) * 255

    b, g, r = cv2.split(result)
    rgba = cv2.merge([b, g, r, alpha])

    return {"rgba": rgba, "result": result, "is_fill": is_fill, "is_colored": is_colored}
