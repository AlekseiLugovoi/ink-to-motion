import os
import io
import json
import base64
import tempfile
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from rembg import remove
from scipy.spatial import Delaunay

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]
CANVAS_H, CANVAS_W = 1240, 1754
MARKER_PX, MARGIN, CONTENT_PAD = 80, 60, 10
CONTENT_SCALE = 0.8

TEMPLATES = {
    "Пустой лист": "assets/template.png",
    "Персонаж 1": "assets/img01_template.png",
    "Персонаж 2": "assets/img02_template.png",
    "Персонаж 3": "assets/img03_template.png",
    "Персонаж 4 (анимация)": "assets/img04_template.png",
    "Персонаж 5 (анимация)": "assets/img05_template.png",
}

ANIM_CHARS = {
    "Персонаж 4 (анимация)": {
        "original": "assets/img04.png",
        "skeleton": "assets/img04_skeleton.json",
        "example_gif": "assets/img04_anim.gif",
        "wave_flip": "ltr",  # переворот слева направо (нос слева)
    },
    "Персонаж 5 (анимация)": {
        "original": "assets/img05.png",
        "skeleton": "assets/img05_skeleton.json",
        "example_gif": "assets/img05_anim.gif",
        "wave_flip": None,  # без переворота
    },
}

FPS = 15
DURATION = 2


# ---------------------------------------------------------------------------
#  Step 1
# ---------------------------------------------------------------------------

def on_select(name):
    path = TEMPLATES[name]
    return path, path


# ---------------------------------------------------------------------------
#  Step 3 — Alignment
# ---------------------------------------------------------------------------

def align(image):
    """Align photo by ArUco markers."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    positions = [
        (MARGIN, MARGIN),
        (CANVAS_W - MARGIN - MARKER_PX, MARGIN),
        (CANVAS_W - MARGIN - MARKER_PX, CANVAS_H - MARGIN - MARKER_PX),
        (MARGIN, CANVAS_H - MARGIN - MARKER_PX),
    ]
    marker_centers = [[px + MARKER_PX / 2, py + MARKER_PX / 2] for px, py in positions]

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT))
    corners, ids, _ = detector.detectMarkers(img)

    if ids is None or len(ids) < 4:
        raise gr.Error(f"Найдено маркеров: {0 if ids is None else len(ids)} из 4. Попробуй переснять.")

    detected = {}
    for i, mid in enumerate(ids.flatten()):
        detected[int(mid)] = corners[i][0].mean(axis=0)

    src_pts = np.array([detected[mid] for mid in MARKER_IDS], dtype=np.float32)
    dst_pts = np.array(marker_centers, dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H), borderValue=(255, 255, 255))

    return Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
#  Step 4 — Crop
# ---------------------------------------------------------------------------

def crop(image):
    """Crop drawing area and remove background."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    inner = MARGIN + MARKER_PX + CONTENT_PAD
    x, y = inner, inner
    w, h = CANVAS_W - 2 * inner, CANVAS_H - 2 * inner
    cropped = img[y : y + h, x : x + w]

    crop_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return remove(crop_pil)


# ---------------------------------------------------------------------------
#  Step 5 — Color transfer + skeleton
# ---------------------------------------------------------------------------

def compute_image_rect(img_path):
    """Compute [x, y, w, h] position of character on template."""
    original = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    ih, iw = original.shape[:2]

    inner = MARGIN + MARKER_PX + CONTENT_PAD
    bx, by = inner, inner
    bw = CANVAS_W - 2 * inner
    bh = CANVAS_H - 2 * inner

    scale = min(bw / iw, bh / ih) * CONTENT_SCALE
    tw, th = int(iw * scale), int(ih * scale)
    x0 = bx + (bw - tw) // 2
    y0 = by + (bh - th) // 2

    return [x0, y0, tw, th]


def transfer_color_app(aligned_pil, char_config):
    """Transfer colors from aligned photo onto original digital character."""
    original_path = char_config["original"]
    image_rect = compute_image_rect(original_path)
    ix, iy, iw, ih = image_rect

    aligned_bgr = cv2.cvtColor(np.array(aligned_pil), cv2.COLOR_RGB2BGR)
    photo_crop = aligned_bgr[iy : iy + ih, ix : ix + iw]

    original = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
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
    is_line = (original_gray < 128).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    is_line_dilated = cv2.dilate(is_line, kernel).astype(bool)
    is_fill = ~is_line_dilated

    photo_hsv = cv2.cvtColor(photo_crop, cv2.COLOR_BGR2HSV)
    is_colored = photo_hsv[:, :, 1] > 40

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

    return {"rgba": rgba, "image_rect": image_rect}


def load_skeleton(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_triangulation(skeleton_data, tw, th):
    skeleton = skeleton_data["skeleton"]
    keypoints = skeleton_data["keypoints"]
    point_names = skeleton["points"]

    skel_pts = np.array(
        [[keypoints[name][0] * tw, keypoints[name][1] * th] for name in point_names],
        dtype=np.float32,
    )
    n_kp = len(point_names)

    corners = np.array([
        [0, 0], [tw, 0], [tw, th], [0, th],
        [tw // 2, 0], [tw, th // 2], [tw // 2, th], [0, th // 2],
    ], dtype=np.float32)

    ctrl_pts = np.vstack([skel_pts, corners])
    tri = Delaunay(ctrl_pts)

    return ctrl_pts, tri, n_kp, point_names


def visualize_skeleton(rgba_img, skeleton_data, ctrl_pts, tri, n_kp):
    """Draw skeleton + triangulation on character image."""
    vis = rgba_img.copy()
    skeleton = skeleton_data["skeleton"]
    point_names = skeleton["points"]
    skel_pts = ctrl_pts[:n_kp]

    for simplex in tri.simplices:
        pts = ctrl_pts[simplex].astype(np.int32)
        cv2.polylines(vis, [pts], True, (200, 100, 0, 255), 1, cv2.LINE_AA)

    for a, b in skeleton["bones"]:
        pa = skel_pts[a].astype(int).tolist()
        pb = skel_pts[b].astype(int).tolist()
        cv2.line(vis, pa, pb, (255, 50, 0, 255), 3, cv2.LINE_AA)

    for i, pt in enumerate(skel_pts):
        center = pt.astype(int).tolist()
        cv2.circle(vis, center, 6, (0, 0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, center, 6, (255, 255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, point_names[i], (center[0] + 10, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0, 255), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGRA2RGBA))


def do_color_transfer_and_skeleton(aligned_img, selected_name):
    """Step 5 handler."""
    if selected_name not in ANIM_CHARS:
        raise gr.Error("Перенос цвета доступен только для персонажей 4 и 5.")
    if aligned_img is None:
        raise gr.Error("Сначала выровняй фото (шаг 3).")

    char_config = ANIM_CHARS[selected_name]
    ct_result = transfer_color_app(aligned_img, char_config)
    rgba = ct_result["rgba"]
    _, _, tw, th = ct_result["image_rect"]

    skel_data = load_skeleton(char_config["skeleton"])
    ctrl_pts, tri, n_kp, _ = build_triangulation(skel_data, tw, th)

    color_pil = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
    skel_pil = visualize_skeleton(rgba, skel_data, ctrl_pts, tri, n_kp)

    return color_pil, skel_pil


# ---------------------------------------------------------------------------
#  Step 6 — Animation
# ---------------------------------------------------------------------------

def make_motion(t, base_pts, n_kp, names):
    pts = base_pts.copy()
    a = t * 2 * np.pi
    idx = {name: i for i, name in enumerate(names)}

    if "tail_base" in idx:
        pts[idx["tail_base"], 1] += np.sin(a) * 8
    if "tail_tip" in idx:
        pts[idx["tail_tip"], 0] += np.sin(a) * 10
        pts[idx["tail_tip"], 1] += np.sin(a) * 20
    if "tail_tip1" in idx:
        pts[idx["tail_tip1"], 0] += np.sin(a) * 10
        pts[idx["tail_tip1"], 1] += np.sin(a) * 15
    if "tail_tip2" in idx:
        pts[idx["tail_tip2"], 0] += np.sin(a) * 10
        pts[idx["tail_tip2"], 1] -= np.sin(a) * 15
    if "top_fin" in idx:
        pts[idx["top_fin"], 0] += np.sin(a + np.pi * 0.5) * 8
        pts[idx["top_fin"], 1] += np.sin(a + np.pi * 0.5) * 5
    if "bottom_fin" in idx:
        pts[idx["bottom_fin"], 1] += np.sin(a - np.pi * 0.5) * 6
    if "nose" in idx:
        pts[idx["nose"], 0] += np.sin(a) * 3
    return pts


def warp_triangle(src_img, src_tri, dst_tri, dst_img):
    r = cv2.boundingRect(np.float32([dst_tri]))
    x, y, w, h = r
    if w <= 0 or h <= 0:
        return
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


def make_wave_flip_frames(img, n_frames=30, direction="ltr"):
    h, w = img.shape[:2]
    cy = h / 2
    cols = np.arange(w, dtype=np.float32)
    dst_ys = np.arange(h, dtype=np.float32)

    if direction == "ltr":
        delays = (cols / w) * 0.6
    else:
        delays = (1 - cols / w) * 0.6

    col_idx = np.broadcast_to(np.arange(w)[None, :], (h, w))

    flip_frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        local_t = np.clip((t - delays) / 0.4, 0, 1)
        scale_y = np.abs(np.cos(local_t * np.pi))

        with np.errstate(divide="ignore", invalid="ignore"):
            src_ys = (cy + (dst_ys[:, None] - cy) / scale_y[None, :]).astype(np.int32)

        valid = (src_ys >= 0) & (src_ys < h) & (scale_y[None, :] >= 0.01)
        src_clamped = np.clip(src_ys, 0, h - 1)

        result = np.zeros_like(img)
        result[valid] = img[src_clamped[valid], col_idx[valid]]
        flip_frames.append(result)
    return flip_frames


def _generate_frames(aligned_img, selected_name):
    """Generate all animation frames (BGRA numpy arrays)."""
    char_config = ANIM_CHARS[selected_name]

    ct_result = transfer_color_app(aligned_img, char_config)
    rgba = ct_result["rgba"]
    _, _, tw, th = ct_result["image_rect"]

    skel_data = load_skeleton(char_config["skeleton"])
    ctrl_pts, tri, n_kp, _ = build_triangulation(skel_data, tw, th)
    point_names = skel_data["skeleton"]["points"]

    if rgba.shape[2] == 3:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2BGRA)

    n_frames = FPS * DURATION
    frames = []
    for i in range(n_frames):
        t = i / n_frames
        dst_pts = make_motion(t, ctrl_pts, n_kp, point_names)
        dst_img = np.zeros_like(rgba)
        for simplex in tri.simplices:
            warp_triangle(rgba, ctrl_pts[simplex].tolist(),
                          dst_pts[simplex].tolist(), dst_img)
        frames.append(dst_img)

    wave_flip = char_config.get("wave_flip")
    if wave_flip:
        flip = make_wave_flip_frames(frames[-1], n_frames=30, direction=wave_flip)
        frames = frames + flip[1:]

    return frames


def _frames_to_gif(pil_frames):
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    pil_frames[0].save(tmp.name, save_all=True, append_images=pil_frames[1:],
                       duration=1000 // FPS, loop=0, disposal=2)
    return tmp.name


def do_animation(aligned_img, selected_name):
    """Step 6 handler — generate animation, cache frames in state."""
    if selected_name not in ANIM_CHARS:
        raise gr.Error("Анимация доступна только для персонажей 4 и 5.")
    if aligned_img is None:
        raise gr.Error("Сначала выровняй фото (шаг 3).")

    all_frames = _generate_frames(aligned_img, selected_name)
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA)) for f in all_frames]
    gif_path = _frames_to_gif(pil_frames)

    return gif_path, all_frames


# ---------------------------------------------------------------------------
#  Step 7 — Composite on background
# ---------------------------------------------------------------------------

BACKGROUND_PATH = "assets/background.jpg"


def do_composite(cached_frames):
    """Step 7 — HTML page: static background + GIF character swimming via CSS."""
    if not cached_frames:
        raise gr.Error("Сначала запусти анимацию (шаг 6).")

    # GIF персонажа из закешированных кадров
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA)) for f in cached_frames]
    gif_buf = io.BytesIO()
    pil_frames[0].save(gif_buf, format="GIF", save_all=True, append_images=pil_frames[1:],
                       duration=1000 // FPS, loop=0, disposal=2)
    gif_b64 = base64.b64encode(gif_buf.getvalue()).decode()

    # фон в base64
    with open(BACKGROUND_PATH, "rb") as f:
        bg_b64 = base64.b64encode(f.read()).decode()

    html = f"""\
<div style="position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;border-radius:12px;cursor:pointer;"
     id="ocean">
  <img src="data:image/jpeg;base64,{bg_b64}"
       style="width:100%;height:100%;object-fit:cover;display:block;">
  <img src="data:image/gif;base64,{gif_b64}" id="fish"
       style="position:absolute;height:35%;
              animation:swimH 8s linear infinite, swimV 3s ease-in-out infinite, tilt 3s ease-in-out infinite;">
</div>
<style>
  @keyframes swimH {{
    from {{ left:100%; }}
    to {{ left:-20%; }}
  }}
  @keyframes swimV {{
    0%,100% {{ top:40%; }}
    50% {{ top:55%; }}
  }}
  @keyframes tilt {{
    0% {{ transform:rotate(0deg); }}
    25% {{ transform:rotate(8deg); }}
    75% {{ transform:rotate(-8deg); }}
    100% {{ transform:rotate(0deg); }}
  }}
</style>
<script>
(function() {{
  var fish = document.getElementById('fish');
  if (!fish) return;
  var speed = 1, target = 1;
  document.getElementById('ocean').addEventListener('click', function() {{
    target = 1.8;
    setTimeout(function() {{ target = 1; }}, 3000);
  }});
  (function loop() {{
    speed += (target - speed) * 0.02;
    fish.style.animationDuration = (8/speed)+'s,'+(3/speed)+'s,'+(3/speed)+'s';
    requestAnimationFrame(loop);
  }})();
}})();
</script>"""

    # сохраняем как файл для скачивания
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
    tmp.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
              f"<title>Ink-to-Motion</title>"
              f"<style>body{{margin:0;background:#000;display:flex;"
              f"align-items:center;justify-content:center;min-height:100vh}}</style>"
              f"</head><body>{html}</body></html>")
    tmp.close()

    return html, tmp.name


# ---------------------------------------------------------------------------
#  Gradio UI
# ---------------------------------------------------------------------------

css = ".gradio-container { max-width: 640px !important; margin: auto !important; }"

with gr.Blocks(title="Ink-to-Motion", theme=gr.themes.Soft(primary_hue="orange"), css=css) as demo:
    gr.Markdown("# Ink-to-Motion")

    with gr.Accordion("Шаг 1. Скачай шаблон", open=True):
        gr.Markdown("Выбери шаблон, распечатай на A4.")
        selector = gr.Dropdown(choices=list(TEMPLATES.keys()), value="Пустой лист", show_label=False)
        preview = gr.Image(value="assets/template.png", show_label=False, interactive=False, height=300)
        download = gr.DownloadButton("Скачать шаблон", value="assets/template.png", variant="secondary")
        selector.change(fn=on_select, inputs=selector, outputs=[preview, download])

    with gr.Accordion("Шаг 2. Сфотографируй рисунок", open=False):
        gr.Markdown("Нарисуй персонажа и покажи что получилось.")
        image_in = gr.Image(show_label=False, type="pil")

    with gr.Accordion("Шаг 3. Выравнивание", open=False):
        align_btn = gr.Button("Выровнять", variant="primary")
        aligned_out = gr.Image(label="Выровненное фото", interactive=False, height=300)
        align_btn.click(fn=align, inputs=image_in, outputs=aligned_out)

    with gr.Accordion("Шаг 4. Вырезка персонажа", open=False):
        crop_btn = gr.Button("Вырезать", variant="primary")
        crop_out = gr.Image(label="Результат", type="pil")
        crop_btn.click(fn=crop, inputs=aligned_out, outputs=crop_out)

    with gr.Accordion("Шаг 5. Перенос цвета + скелет", open=False):
        gr.Markdown("Цвета с фото переносятся на оригинального персонажа. Доступно для персонажей 4 и 5.")
        ct_btn = gr.Button("Перенести цвет", variant="primary")
        color_out = gr.Image(label="Перенос цвета", interactive=False, height=300)
        skel_out = gr.Image(label="Скелет + триангуляция", interactive=False, height=300)
        ct_btn.click(fn=do_color_transfer_and_skeleton, inputs=[aligned_out, selector], outputs=[color_out, skel_out])

    frames_state = gr.State([])

    with gr.Accordion("Шаг 6. Анимация", open=False):
        gr.Markdown("Посмотри как персонаж оживает! Референс: [teamLab Sketch Ocean](https://www.teamlab.art/w/sketch_ocean/)")
        with gr.Accordion("Примеры анимации", open=False):
            with gr.Row():
                gr.Image(value="assets/img04_anim.gif", label="Персонаж 4", interactive=False, height=200)
                gr.Image(value="assets/img05_anim.gif", label="Персонаж 5", interactive=False, height=200)
        anim_btn = gr.Button("Анимировать", variant="primary")
        anim_out = gr.Image(label="Анимация")
        anim_btn.click(fn=do_animation, inputs=[aligned_out, selector], outputs=[anim_out, frames_state])

    with gr.Accordion("Шаг 7. На фоне", open=False):
        gr.Markdown("Персонаж проплывает по подводному фону (CSS-анимация).")
        composite_btn = gr.Button("Поместить на фон", variant="primary")
        composite_out = gr.HTML(label="Результат")
        composite_file = gr.File(label="Скачать HTML")
        composite_btn.click(fn=do_composite, inputs=frames_state, outputs=[composite_out, composite_file])

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
