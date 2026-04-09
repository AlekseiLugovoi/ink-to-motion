import io
import os
import base64
import tempfile
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from scipy.spatial import Delaunay

try:
    from .config import (
        ARUCO_DICT, MARKER_IDS,
        CANVAS_H, CANVAS_W,
        MARKER_PX, MARGIN, CONTENT_PAD, CONTENT_SCALE,
        FPS, DURATION,
        BACKGROUND_PATH, BACKGROUND_VIDEO_PATH,
        CHARS,
    )
except ImportError:
    from config import (
        ARUCO_DICT, MARKER_IDS,
        CANVAS_H, CANVAS_W,
        MARKER_PX, MARGIN, CONTENT_PAD, CONTENT_SCALE,
        FPS, DURATION,
        BACKGROUND_PATH, BACKGROUND_VIDEO_PATH,
        CHARS,
    )

# ---------------------------------------------------------------------------
#  Camera helpers
# ---------------------------------------------------------------------------

def decode_camera_photo(b64_str):
    if not b64_str:
        return None
    _, data = b64_str.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


_aruco_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
)


def detect_aruco(frame_b64):
    if not frame_b64 or "," not in frame_b64:
        return "0"
    try:
        _, data = frame_b64.split(",", 1)
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(data), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if img is None:
            return "0"
        _, ids, _ = _aruco_detector.detectMarkers(img)
        if ids is None:
            return "0"
        return str(len(set(ids.flatten().tolist()) & {0, 1, 2, 3}))
    except Exception:
        return "0"


# ---------------------------------------------------------------------------
#  Alignment
# ---------------------------------------------------------------------------

def _marker_centers():
    positions = [
        (MARGIN, MARGIN),
        (CANVAS_W - MARGIN - MARKER_PX, MARGIN),
        (CANVAS_W - MARGIN - MARKER_PX, CANVAS_H - MARGIN - MARKER_PX),
        (MARGIN, CANVAS_H - MARGIN - MARKER_PX),
    ]
    return [[px + MARKER_PX / 2, py + MARKER_PX / 2] for px, py in positions]


def align(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    )
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(ids) < 4:
        raise gr.Error(
            f"На фото найдено "
            f"{0 if ids is None else len(ids)} маркеров из 4. "
            "Покажи в кадре все четыре ArUco-маркера."
        )
    detected = {}
    for i, mid in enumerate(ids.flatten()):
        detected[int(mid)] = corners[i][0].mean(axis=0)
    src_pts = np.array([detected[mid] for mid in MARKER_IDS], dtype=np.float32)
    dst_pts = np.array(_marker_centers(), dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(
        img, M, (CANVAS_W, CANVAS_H), borderValue=(255, 255, 255)
    )
    return Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
#  Color transfer (by mask)
# ---------------------------------------------------------------------------

def compute_image_rect(img_path):
    original = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    ih, iw = original.shape[:2]
    inner = MARGIN + MARKER_PX + CONTENT_PAD
    bx, by = inner, inner
    bw, bh = CANVAS_W - 2 * inner, CANVAS_H - 2 * inner
    scale = min(bw / iw, bh / ih) * CONTENT_SCALE
    tw, th = int(iw * scale), int(ih * scale)
    return [bx + (bw - tw) // 2, by + (bh - th) // 2, tw, th]


def transfer_color_app(aligned_pil, char):
    """Transfer colors from aligned photo onto character using mask.
    Green (0,255,0) in mask = fill zone from photo.
    Transparent = discard (background).
    Everything else = keep from original.
    """
    image_rect = compute_image_rect(char["drawing"])
    ix, iy, iw, ih = image_rect

    aligned_bgr = cv2.cvtColor(np.array(aligned_pil), cv2.COLOR_RGB2BGR)
    photo_crop = aligned_bgr[iy : iy + ih, ix : ix + iw]

    original = cv2.imread(char["drawing"], cv2.IMREAD_UNCHANGED)
    original = cv2.resize(original, (iw, ih), interpolation=cv2.INTER_AREA)

    mask_img = cv2.imread(char["mask"], cv2.IMREAD_UNCHANGED)
    mask_img = cv2.resize(mask_img, (iw, ih), interpolation=cv2.INTER_NEAREST)

    if mask_img.shape[2] == 4:
        is_fill = (
            (mask_img[:, :, 0] == 0)
            & (mask_img[:, :, 1] == 255)
            & (mask_img[:, :, 2] == 0)
            & (mask_img[:, :, 3] > 0)
        )
        final_alpha = mask_img[:, :, 3].copy()
    else:
        is_fill = (
            (mask_img[:, :, 0] == 0)
            & (mask_img[:, :, 1] == 255)
            & (mask_img[:, :, 2] == 0)
        )
        final_alpha = (mask_img.mean(axis=2) < 250).astype(np.uint8) * 255

    clean = cv2.fastNlMeansDenoisingColored(photo_crop, None, 8, 8, 7, 21)
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    A = cv2.GaussianBlur(A, (0, 0), 3.0)
    B = cv2.GaussianBlur(B, (0, 0), 3.0)
    clean = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    if original.shape[2] == 4:
        a_ch = original[:, :, 3:4] / 255.0
        result = (original[:, :, :3] * a_ch + 255 * (1 - a_ch)).astype(np.uint8)
    else:
        result = original.copy()

    result[is_fill] = clean[is_fill]
    rgba = cv2.merge([*cv2.split(result), final_alpha])
    return {"rgba": rgba, "image_rect": image_rect}


# ---------------------------------------------------------------------------
#  Triangulation + Animation
# ---------------------------------------------------------------------------

def build_triangulation(keypoints, tw, th):
    names = list(keypoints.keys())
    skel_pts = np.array(
        [[kp[0] * tw, kp[1] * th] for kp in keypoints.values()],
        dtype=np.float32,
    )
    corners = np.array(
        [
            [0, 0], [tw, 0], [tw, th], [0, th],
            [tw // 2, 0], [tw, th // 2], [tw // 2, th], [0, th // 2],
        ],
        dtype=np.float32,
    )
    ctrl_pts = np.vstack([skel_pts, corners])
    tri = Delaunay(ctrl_pts)
    return ctrl_pts, tri, len(skel_pts), names


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
    warped = cv2.warpAffine(
        src_img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_local), 255)
    roi = dst_img[y:y2, x:x2]
    m = mask[: roi.shape[0], : roi.shape[1]] > 0
    warped = warped[: roi.shape[0], : roi.shape[1]]
    for c in range(dst_img.shape[2]):
        roi[:, :, c] = np.where(m, warped[:, :, c], roi[:, :, c])


def _generate_frames(aligned_img, char):
    ct_result = transfer_color_app(aligned_img, char)
    rgba = ct_result["rgba"]
    _, _, tw, th = ct_result["image_rect"]

    keypoints = char["skeleton"]["keypoints"]
    motion_cfg = char["motion"]
    ctrl_pts, tri, n_kp, names = build_triangulation(keypoints, tw, th)

    if rgba.shape[2] == 3:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2BGRA)

    n_frames = FPS * DURATION
    frames = []
    for i in range(n_frames):
        t = i / n_frames
        dst_pts = make_motion_from_config(t, ctrl_pts, n_kp, names, motion_cfg)
        dst_img = np.zeros_like(rgba)
        for simplex in tri.simplices:
            warp_triangle(
                rgba, ctrl_pts[simplex].tolist(),
                dst_pts[simplex].tolist(), dst_img,
            )
        frames.append(dst_img)
    return frames


# ---------------------------------------------------------------------------
#  GIF helpers
# ---------------------------------------------------------------------------

def _rgba_to_gif_frame(frame_rgba):
    alpha = np.array(frame_rgba.split()[3]) < 128
    frame_p = frame_rgba.convert("RGB").quantize(colors=255)
    palette = frame_p.getpalette()
    palette[255 * 3 : 255 * 3 + 3] = [0, 0, 0]
    indices = np.array(frame_p, dtype=np.uint8)
    indices[alpha] = 255
    result = Image.fromarray(indices, mode="P")
    result.putpalette(palette)
    result.info["transparency"] = 255
    return result


def _frames_to_gif(pil_frames):
    gif_frames = [_rgba_to_gif_frame(f) for f in pil_frames]
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    gif_frames[0].save(
        tmp.name, save_all=True, append_images=gif_frames[1:],
        duration=1000 // FPS, loop=0, disposal=2, transparency=255,
    )
    return tmp.name


# ---------------------------------------------------------------------------
#  Composite on background
# ---------------------------------------------------------------------------

def _composite_html(cached_frames):
    pil_frames = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA))
        for f in cached_frames
    ]
    gif_frames = [_rgba_to_gif_frame(f) for f in pil_frames]
    gif_buf = io.BytesIO()
    gif_frames[0].save(
        gif_buf, format="GIF", save_all=True, append_images=gif_frames[1:],
        duration=1000 // FPS, loop=0, disposal=2, transparency=255,
    )
    gif_b64 = base64.b64encode(gif_buf.getvalue()).decode()

    background_markup = ""
    if os.path.exists(BACKGROUND_VIDEO_PATH):
        with open(BACKGROUND_VIDEO_PATH, "rb") as f:
            bg_video_b64 = base64.b64encode(f.read()).decode()
        background_markup = f"""
  <video autoplay muted loop playsinline
         style="width:100%;height:100%;object-fit:cover;display:block;">
    <source src="data:video/mp4;base64,{bg_video_b64}" type="video/mp4">
  </video>"""
    else:
        with open(BACKGROUND_PATH, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
        background_markup = f"""
  <img src="data:image/jpeg;base64,{bg_b64}"
       style="width:100%;height:100%;object-fit:cover;display:block;">"""

    html = f"""\
<div style="position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;border-radius:12px;cursor:pointer;"
     id="ocean">
  {background_markup}
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
    return html


# ---------------------------------------------------------------------------
#  Full pipeline (auto mode)
# ---------------------------------------------------------------------------

def auto_process(photo, char_id):
    """Run align -> animate -> composite. Returns (gif_path, composite_html)."""
    if photo is None:
        return None, ""
    char = CHARS.get(char_id)
    if not char:
        raise gr.Error("Персонаж не найден.")

    aligned = align(photo)
    frames = _generate_frames(aligned, char)
    pil_frames = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA))
        for f in frames
    ]
    gif_path = _frames_to_gif(pil_frames)
    html = _composite_html(frames)
    return gif_path, html


# ---------------------------------------------------------------------------
#  Step-by-step handlers
# ---------------------------------------------------------------------------

def on_select_char(char_id):
    char = CHARS.get(char_id)
    if not char:
        return None, None
    return char.get("template"), char.get("preview")


def _render_qr_html(char_id):
    if not char_id or char_id not in CHARS:
        return ""
    return (
        f'<div class="qr-center-wrap">'
        f'  <div data-qr-char="{char_id}" style="padding:8px;"></div>'
        f'</div>'
    )


def _character_preset_values(char_id):
    char = CHARS.get(char_id)
    if not char:
        return None, None, None, None
    return (
        char.get("drawing"),
        char.get("mask"),
        char.get("skeleton_png"),
        char.get("preview"),
    )


def _on_auto_char_change(char_id):
    if not char_id:
        return "", None, None, ""
    char = CHARS.get(char_id)
    if not char:
        return "", None, None, ""
    template = char.get("template")
    return char_id, template, template, _render_qr_html(char_id)


def _on_step_char_change(char_id):
    char = CHARS.get(char_id)
    if not char:
        return None, None, None, None, None, None, ""
    template = char.get("template")
    drawing, mask, skeleton_png, preview = _character_preset_values(char_id)
    return drawing, mask, skeleton_png, preview, template, template, _render_qr_html(char_id)


def do_color_transfer(aligned_img, char_id):
    if aligned_img is None:
        raise gr.Error("Сначала загрузи и выровняй фото рисунка.")
    char = CHARS.get(char_id)
    if not char:
        raise gr.Error("Персонаж не найден.")
    ct_result = transfer_color_app(aligned_img, char)
    rgba = ct_result["rgba"]
    return Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))


def do_animation(aligned_img, char_id):
    if aligned_img is None:
        raise gr.Error("Сначала загрузи и выровняй фото рисунка.")
    char = CHARS.get(char_id)
    if not char:
        raise gr.Error("Персонаж не найден.")
    frames = _generate_frames(aligned_img, char)
    pil_frames = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA))
        for f in frames
    ]
    gif_path = _frames_to_gif(pil_frames)
    return gif_path, frames


def do_composite(cached_frames):
    if not cached_frames:
        raise gr.Error("Сначала собери анимацию.")
    html = _composite_html(cached_frames)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8",
    )
    tmp.write(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Ink-to-Motion</title>"
        f"<style>body{{margin:0;background:#000;display:flex;"
        f"align-items:center;justify-content:center;min-height:100vh}}</style>"
        f"</head><body>{html}</body></html>"
    )
    tmp.close()
    return html, tmp.name
