import io
import os
import base64
import random
import tempfile
import time
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from xml.etree import ElementTree as ET

try:
    from .config import (
        ARUCO_DICT, MARKER_IDS,
        CANVAS_H, CANVAS_W,
        MARKER_PX, MARGIN, CONTENT_PAD, CONTENT_SCALE,
        FPS, DURATION,
        BACKGROUND_PATH, BACKGROUND_VIDEO_PATH,
        CHARS,
    )
    from .arap import animate_arap as _animate_arap
except ImportError:
    from config import (
        ARUCO_DICT, MARKER_IDS,
        CANVAS_H, CANVAS_W,
        MARKER_PX, MARGIN, CONTENT_PAD, CONTENT_SCALE,
        FPS, DURATION,
        BACKGROUND_PATH, BACKGROUND_VIDEO_PATH,
        CHARS,
    )
    from arap import animate_arap as _animate_arap

# ---------------------------------------------------------------------------
#  Temp file tracking & cleanup
# ---------------------------------------------------------------------------

_temp_files: list[str] = []


def _track_temp(path: str) -> str:
    """Register a temp file for later cleanup."""
    _temp_files.append(path)
    return path


def _cleanup_temp():
    """Delete all tracked temp files."""
    for p in _temp_files:
        try:
            os.remove(p)
        except OSError:
            pass
    _temp_files.clear()


import atexit
atexit.register(_cleanup_temp)

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

def compute_image_rect(svg_path):
    """Compute image_rect from SVG dimensions + template geometry."""
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    svg_w = float(root.get("width"))
    svg_h = float(root.get("height"))
    inner = MARGIN + MARKER_PX + CONTENT_PAD
    bx, by = inner, inner
    bw, bh = CANVAS_W - 2 * inner, CANVAS_H - 2 * inner
    scale = min(bw / svg_w, bh / svg_h) * CONTENT_SCALE
    tw, th = int(svg_w * scale), int(svg_h * scale)
    return [bx + (bw - tw) // 2, by + (bh - th) // 2, tw, th]


def _correct_photo(aligned_bgr, image_rect, white_target=240, gain_range=(0.9, 1.3)):
    """White balance correction using paper margins as reference."""
    ix, iy, iw, ih = image_rect
    inner = MARGIN + MARKER_PX + CONTENT_PAD
    bx, by = inner, inner
    bw, bh = CANVAS_W - 2 * inner, CANVAS_H - 2 * inner

    strips = []
    if iy > by:       strips.append(aligned_bgr[by:iy, bx:bx+bw])
    if iy+ih < by+bh: strips.append(aligned_bgr[iy+ih:by+bh, bx:bx+bw])
    if ix > bx:       strips.append(aligned_bgr[by:by+bh, bx:ix])
    if ix+iw < bx+bw: strips.append(aligned_bgr[by:by+bh, ix+iw:bx+bw])

    paper_pixels = np.vstack([s.reshape(-1, 3) for s in strips if s.size > 0])
    paper_ref = np.median(paper_pixels, axis=0)
    gain = np.clip(white_target / (paper_ref + 1e-6), *gain_range)

    return np.clip(aligned_bgr.astype(np.float32) * gain[np.newaxis, np.newaxis, :], 0, 255).astype(np.uint8)


_SVG_COLOR_MAP = {
    "#00ff00": (0, 255, 0), "#00FF00": (0, 255, 0),
    "black": (0, 0, 0), "#000000": (0, 0, 0), "#000": (0, 0, 0),
    "white": (255, 255, 255), "#ffffff": (255, 255, 255), "#FFFFFF": (255, 255, 255),
}


def _svg_parse_paths(svg_path, sx, sy, pts_per_seg=30):
    """Parse SVG paths into ordered draw operations (painter's algorithm).

    Returns list of (polygon_pts, fill_bgr, stroke_bgr, thickness).
    """
    from svgpathtools import parse_path

    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ns = "http://www.w3.org/2000/svg"

    ops = []
    for path_el in root.iter(f"{{{ns}}}path"):
        d = (path_el.get("d") or "").strip()
        if not d:
            continue
        fill_bgr = _SVG_COLOR_MAP.get((path_el.get("fill") or "none").strip())
        stroke_bgr = _SVG_COLOR_MAP.get((path_el.get("stroke") or "none").strip())

        path_obj = parse_path(d)
        subpaths, current = [], []
        for seg in path_obj:
            start = (seg.start.real * sx, seg.start.imag * sy)
            if current:
                last = current[-1]
                if (start[0] - last[0]) ** 2 + (start[1] - last[1]) ** 2 > 4:
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

        for pts in subpaths:
            ops.append((pts, fill_bgr, stroke_bgr, 1))
    return ops


def _render_svg_polygons(svg_path, tw, th):
    """Render SVG character using polygons. Returns BGRA numpy array."""
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    svg_w = float(root.get("width"))
    svg_h = float(root.get("height"))

    ops = _svg_parse_paths(svg_path, tw / svg_w, th / svg_h)
    green_bgr = (0, 255, 0)

    result = np.full((th, tw, 3), 255, dtype=np.uint8)
    alpha = np.zeros((th, tw), dtype=np.uint8)

    for pts, fill_bgr, _, _ in ops:
        if fill_bgr == green_bgr:
            cv2.fillPoly(alpha, [pts], 255)
        elif fill_bgr:
            cv2.fillPoly(result, [pts], fill_bgr)
            cv2.fillPoly(alpha, [pts], 255)
    for pts, _, stroke_bgr, thickness in ops:
        if stroke_bgr:
            cv2.polylines(result, [pts], False, stroke_bgr, thickness)
            cv2.polylines(alpha, [pts], False, 255, thickness)

    return cv2.merge([*cv2.split(result), alpha])


def transfer_color_app(aligned_pil, char):
    """Transfer colors from photo onto SVG character using polygon approach.

    SVG paths -> cv2.fillPoly/polylines. Painter's algorithm:
    1. Green fills -> photo color zone
    2. Non-green fills -> overlay (eye, pupil, etc.)
    3. Strokes -> contours on top
    """
    svg_path = char["svg"]
    image_rect = compute_image_rect(svg_path)
    ix, iy, iw, ih = image_rect

    aligned_bgr = cv2.cvtColor(np.array(aligned_pil), cv2.COLOR_RGB2BGR)
    aligned_bgr = _correct_photo(aligned_bgr, image_rect)
    photo_crop = aligned_bgr[iy : iy + ih, ix : ix + iw]

    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    svg_w = float(root.get("width"))
    svg_h = float(root.get("height"))

    ops = _svg_parse_paths(svg_path, iw / svg_w, ih / svg_h)
    green_bgr = (0, 255, 0)

    result = photo_crop.copy()
    final_alpha = np.zeros((ih, iw), dtype=np.uint8)

    for pts, fill_bgr, _, _ in ops:
        if fill_bgr == green_bgr:
            cv2.fillPoly(final_alpha, [pts], 255)
        elif fill_bgr:
            cv2.fillPoly(result, [pts], fill_bgr)
            cv2.fillPoly(final_alpha, [pts], 255)

    for pts, _, stroke_bgr, thickness in ops:
        if stroke_bgr:
            cv2.polylines(result, [pts], False, stroke_bgr, thickness)
            cv2.polylines(final_alpha, [pts], False, 255, thickness)

    # Закрыть 1px дырки между полигонами (close = dilate + erode)
    final_alpha = cv2.morphologyEx(final_alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    rgba = cv2.merge([*cv2.split(result), final_alpha])
    return {"rgba": rgba, "image_rect": image_rect}


def _generate_frames(aligned_img, char):
    ct_result = transfer_color_app(aligned_img, char)
    rgba = ct_result["rgba"]
    keypoints = char["skeleton"]["keypoints"]
    animation_cfg = char["animation"]
    return _animate_arap(rgba, keypoints, animation_cfg,
                         fps=FPS, duration=DURATION, pad=(40, 60, 40, 40))


# ---------------------------------------------------------------------------
#  Video encoding
# ---------------------------------------------------------------------------

def _frames_to_mp4(cv_frames):
    """Encode BGRA frames to H.264 mp4 (white background for transparency)."""
    import imageio_ffmpeg
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    h, w = cv_frames[0].shape[:2]
    # Pad to multiples of 16 (H.264 macro_block_size requirement)
    w16 = (w + 15) // 16 * 16
    h16 = (h + 15) // 16 * 16
    need_pad = (w16 != w or h16 != h)
    writer = imageio_ffmpeg.write_frames(
        tmp.name, (w16, h16), fps=FPS,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "fast", "-movflags", "+faststart"],
    )
    writer.send(None)  # init
    for frame in cv_frames:
        if frame.shape[2] == 4:
            alpha = frame[:, :, 3:4] / 255.0
            rgb = cv2.cvtColor(
                (frame[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8),
                cv2.COLOR_BGR2RGB,
            )
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if need_pad:
            rgb = cv2.copyMakeBorder(rgb, 0, h16 - h, 0, w16 - w,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))
        writer.send(rgb.tobytes())
    writer.close()
    return _track_temp(tmp.name)


def _frames_to_apng(cv_frames, max_h=300):
    """Encode BGRA frames to APNG (animated PNG) with full alpha support."""
    h, w = cv_frames[0].shape[:2]
    scale = min(1.0, max_h / h)
    pil_frames = []
    for f in cv_frames:
        rgba = cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(rgba)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        pil_frames.append(img)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    pil_frames[0].save(
        tmp.name, format="PNG", save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // FPS, loop=0,
    )
    return _track_temp(tmp.name)


# ---------------------------------------------------------------------------
#  Composite on background
# ---------------------------------------------------------------------------

def _composite_html(cached_frames, swim_cfg=None):
    fish_path = _frames_to_apng(cached_frames)
    with open(fish_path, "rb") as f:
        fish_b64 = base64.b64encode(f.read()).decode()

    if swim_cfg is None:
        swim_cfg = {}
    direction = swim_cfg.get("direction", 180)
    flip = swim_cfg.get("flip", False)
    rot = swim_cfg.get("rotate", 0)
    swim_dur = swim_cfg.get("swim_duration", 8)
    bob = swim_cfg.get("bob_range", [40, 55])
    tilt_amp = swim_cfg.get("tilt", 8)
    tilt_dur = swim_cfg.get("tilt_duration", 3)
    offset_pct = swim_cfg.get("offset_x", 0)

    total_dur = swim_dur * 2  # round trip

    # Bounce points (visible, not off-screen)
    dir_rad = np.radians(direction)
    cos_d, sin_d = np.cos(dir_rad), np.sin(dir_rad)
    span = 35
    ax = 50 - cos_d * span + offset_pct
    ay = 50 + sin_d * span
    bx = 50 + cos_d * span + offset_pct
    by = 50 - sin_d * span
    bob_amp = (bob[1] - bob[0]) / 2

    # Forward / return orientation
    sx_fwd = -1 if flip else 1
    sx_ret = -sx_fwd

    # Bake round-trip path + bob + tilt + turn into one keyframe
    turn = 0.025  # 2.5% of animation for turn (~0.4s)
    fwd_end = 0.5 - turn
    ret_start = 0.5 + turn
    n_steps = 80
    kf_lines = []
    for i in range(n_steps + 1):
        t = i / n_steps
        if t <= fwd_end:
            frac = t / fwd_end
            lv = ax + (bx - ax) * frac
            tv = ay + (by - ay) * frac
            phase = frac * swim_dur / tilt_dur
            tv += bob_amp * np.sin(2 * np.pi * phase)
            tilt_v = tilt_amp * np.sin(2 * np.pi * phase)
            r = -rot - tilt_v
            sx = sx_fwd
        elif t <= 0.5:
            lv, tv = bx, by
            squeeze = 1 - (t - fwd_end) / turn
            r = -rot
            sx = sx_fwd * squeeze
        elif t <= ret_start:
            lv, tv = bx, by
            expand = (t - 0.5) / turn
            r = rot
            sx = sx_ret * expand
        else:
            frac = (t - ret_start) / (1 - ret_start)
            lv = bx + (ax - bx) * frac
            tv = by + (ay - by) * frac
            phase = frac * swim_dur / tilt_dur
            tv += bob_amp * np.sin(2 * np.pi * phase)
            tilt_v = tilt_amp * np.sin(2 * np.pi * phase)
            r = rot - tilt_v
            sx = sx_ret
        tf = f"translate(-50%,-50%) rotate({r:.1f}deg) scaleX({sx:.3f})"
        kf_lines.append(f"    {t*100:.1f}%{{left:{lv:.1f}%;top:{tv:.1f}%;transform:{tf};}}")
    swim_css = "\n".join(kf_lines)

    background_markup = ""
    if os.path.exists(BACKGROUND_VIDEO_PATH):
        background_markup = (
            '<video autoplay muted loop playsinline'
            ' style="width:100%;height:100%;object-fit:cover;display:block;">'
            f'<source src="/gradio_api/file={BACKGROUND_VIDEO_PATH}">'
            '</video>')
    else:
        background_markup = (
            f'<img src="/gradio_api/file={BACKGROUND_PATH}"'
            ' style="width:100%;height:100%;object-fit:cover;display:block;">')

    html = f"""\
<div style="position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;border-radius:12px;cursor:pointer;"
     id="ocean">
  {background_markup}
  <img id="fish"
       src="data:image/png;base64,{fish_b64}"
       style="position:absolute;height:35%;
              animation:swim {total_dur}s linear infinite;">
</div>
<style>
  @keyframes swim {{
{swim_css}
  }}
</style>
<script>
(function() {{
  var fish = document.getElementById('fish');
  if (!fish) return;
  var speed = 1, target = 1, rafId = 0;
  function loop() {{
    speed += (target - speed) * 0.002;
    fish.style.animationDuration = ({total_dur}/speed)+'s';
    if (Math.abs(speed - target) > 0.001) {{
      rafId = requestAnimationFrame(loop);
    }}
  }}
  document.getElementById('ocean').addEventListener('click', function() {{
    target = 1.8;
    setTimeout(function() {{ target = 1; cancelAnimationFrame(rafId); loop(); }}, 3000);
    cancelAnimationFrame(rafId);
    loop();
  }});
}})();
</script>"""
    return html


# ---------------------------------------------------------------------------
#  Aquarium
# ---------------------------------------------------------------------------

AQUARIUM_EMPTY = (
    '<div style="min-height:300px;display:flex;align-items:center;'
    'justify-content:center;color:#9ca3af;border:1px dashed #d1d5db;'
    'border-radius:12px;background:#f8fafc;">'
    '<span style="font-size:18px;">Аквариум пуст — добавь персонажа</span></div>'
)


def _swim_keyframes(swim_cfg, name="swim", offset_y=0):
    """Generate CSS @keyframes + total_dur for one fish."""
    cfg = swim_cfg or {}
    direction = cfg.get("direction", 180)
    flip = cfg.get("flip", False)
    rot = cfg.get("rotate", 0)
    swim_dur = cfg.get("swim_duration", 8)
    bob = cfg.get("bob_range", [40, 55])
    tilt_amp = cfg.get("tilt", 8)
    tilt_dur = cfg.get("tilt_duration", 3)
    offset_pct = cfg.get("offset_x", 0)

    total_dur = swim_dur * 2
    dir_rad = np.radians(direction)
    cos_d, sin_d = np.cos(dir_rad), np.sin(dir_rad)
    span = 35
    ax = 50 - cos_d * span + offset_pct
    ay = 50 + sin_d * span + offset_y
    bx = 50 + cos_d * span + offset_pct
    by = 50 - sin_d * span + offset_y
    bob_amp = (bob[1] - bob[0]) / 2 if isinstance(bob, (list, tuple)) and len(bob) >= 2 else 5
    tilt_dur = tilt_dur or 3  # guard against 0 / None
    sx_fwd = -1 if flip else 1
    sx_ret = -sx_fwd
    turn = 0.025
    fwd_end = 0.5 - turn
    ret_start = 0.5 + turn

    kf_lines = []
    n_steps = 80
    for i in range(n_steps + 1):
        t = i / n_steps
        if t <= fwd_end:
            frac = t / fwd_end
            lv = ax + (bx - ax) * frac
            tv = ay + (by - ay) * frac
            phase = frac * swim_dur / tilt_dur
            tv += bob_amp * np.sin(2 * np.pi * phase)
            tilt_v = tilt_amp * np.sin(2 * np.pi * phase)
            r, sx = -rot - tilt_v, sx_fwd
        elif t <= 0.5:
            lv, tv = bx, by
            r, sx = -rot, sx_fwd * (1 - (t - fwd_end) / turn)
        elif t <= ret_start:
            lv, tv = bx, by
            r, sx = rot, sx_ret * ((t - 0.5) / turn)
        else:
            frac = (t - ret_start) / (1 - ret_start)
            lv = bx + (ax - bx) * frac
            tv = by + (ay - by) * frac
            phase = frac * swim_dur / tilt_dur
            tv += bob_amp * np.sin(2 * np.pi * phase)
            tilt_v = tilt_amp * np.sin(2 * np.pi * phase)
            r, sx = rot - tilt_v, sx_ret
        # Guard against NaN from bad config values
        if np.isnan(lv) or np.isnan(tv) or np.isnan(r) or np.isnan(sx):
            print(f"[WARN] NaN at step {i}/{n_steps} in {name}: "
                  f"lv={lv}, tv={tv}, r={r}, sx={sx}, cfg={cfg}")
            lv = np.nan_to_num(lv, nan=50.0)
            tv = np.nan_to_num(tv, nan=50.0)
            r = np.nan_to_num(r, nan=0.0)
            sx = np.nan_to_num(sx, nan=1.0)
        tf = f"translate(-50%,-50%) rotate({r:.1f}deg) scaleX({sx:.3f})"
        kf_lines.append(
            f"{t*100:.1f}% {{ left:{lv:.1f}%; top:{tv:.1f}%; transform:{tf}; }}")

    css = f"@keyframes {name} {{\n" + "\n".join(kf_lines) + "\n}"
    return css, total_dur


def _aquarium_html(fish_list):
    """Generate aquarium HTML with inline <style> and unique keyframe names."""
    if not fish_list:
        return AQUARIUM_EMPTY, ""

    # Unique prefix to avoid stale CSS keyframe collisions across updates
    uid = int(time.time() * 1000) % 1_000_000

    bg_path = BACKGROUND_VIDEO_PATH if os.path.exists(BACKGROUND_VIDEO_PATH) else BACKGROUND_PATH
    if os.path.exists(BACKGROUND_VIDEO_PATH):
        bg = (
            '<video autoplay muted loop playsinline'
            ' style="width:100%;height:100%;object-fit:cover;display:block;">'
            f'<source src="/gradio_api/file={BACKGROUND_VIDEO_PATH}">'
            '</video>')
    else:
        bg = (
            f'<img src="/gradio_api/file={BACKGROUND_PATH}"'
            ' style="width:100%;height:100%;object-fit:cover;display:block;">')

    imgs = []
    styles = []
    for idx, fish in enumerate(fish_list):
        kf_name = f"aq{uid}_{idx}"
        cfg = fish["swim_cfg"] or {}
        oy = cfg.get("offset_y", fish.get("offset_y", 0))
        css, total_dur = _swim_keyframes(cfg, name=kf_name, offset_y=oy)
        styles.append(css)
        delay = cfg.get("delay", fish.get("delay", 0))
        size = cfg.get("size", fish.get("size", 30))
        apng_path = fish["apng_path"].replace("\\", "/")
        imgs.append(
            f'<img src="/gradio_api/file={apng_path}"'
            f' style="position:absolute;height:{size}%;z-index:{idx + 1};'
            f'animation:{kf_name} {total_dur}s linear infinite -{delay:.1f}s;">')
        print(f"[aquarium] fish #{idx}: {kf_name} dur={total_dur}s delay=-{delay}s")

    all_css = "\n".join(styles)

    # Save standalone fullscreen HTML
    page = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>*{{margin:0;padding:0;}}body{{overflow:hidden;}}\n{all_css}\n</style>"
            f"</head><body>"
            f"<div style='position:relative;width:100vw;height:100vh;overflow:hidden;'>"
            f"{bg}{''.join(imgs)}</div></body></html>")
    fullscreen_file = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8")
    fullscreen_file.write(page)
    fullscreen_file.close()
    _track_temp(fullscreen_file.name)
    fs_url = f"/gradio_api/file={fullscreen_file.name.replace(chr(92), '/')}"

    display = (
        f'<div style="position:relative;width:100%;aspect-ratio:16/9;'
        f'overflow:hidden;border-radius:12px;">'
        f'{bg}'
        f'{"".join(imgs)}'
        f'</div>'
        f'\n<style>\n{all_css}\n</style>'
    )

    return display, fs_url


def add_to_aquarium(last_fish, aquarium_state):
    """Add the last processed fish to the aquarium."""
    if not last_fish:
        raise gr.Error("Сначала обработай фото во вкладке Авто.")
    aquarium_state = list(aquarium_state or [])
    if len(aquarium_state) >= 5:
        raise gr.Error("Максимум 5 персонажей. Нажми «Очистить» чтобы начать заново.")
    # Pick a pattern that hasn't been used yet (if possible)
    used_dirs = {f["swim_cfg"].get("direction") for f in aquarium_state}
    available = [p for p in last_fish["patterns"] if p.get("direction") not in used_dirs]
    swim_cfg = random.choice(available) if available else random.choice(last_fish["patterns"])

    total_dur = swim_cfg.get("swim_duration", 8) * 2
    # Fixed vertical offsets to guarantee separation
    y_slots = [-20, 15, -8, 22, -15]
    idx = len(aquarium_state)
    fish = {
        "apng_path": last_fish["apng_path"],
        "swim_cfg": swim_cfg,
        "delay": round(total_dur * (idx * 0.3 % 1), 1),
        "size": [30, 28, 33, 25, 35][idx % 5],
        "offset_y": y_slots[idx % 5],
    }
    aquarium_state.append(fish)
    print(f"[aquarium] adding fish #{idx+1}: swim_cfg={swim_cfg}")
    display, btn = _aquarium_html(aquarium_state)
    return aquarium_state, display, btn


def clear_aquarium():
    """Remove all fish from the aquarium."""
    return [], AQUARIUM_EMPTY, ""


# ---------------------------------------------------------------------------
#  Full pipeline (auto mode)
# ---------------------------------------------------------------------------

def auto_process(photo, char_id):
    """Run align -> animate -> composite. Returns (anim_html, composite_html, fish_data)."""
    if photo is None:
        return "", "", None
    char = CHARS.get(char_id)
    if not char:
        raise gr.Error("Персонаж не найден.")

    aligned = align(photo)
    frames = _generate_frames(aligned, char)
    motion_patterns = char.get("motion", {})
    pattern = random.choice(list(motion_patterns.values())) if motion_patterns else None
    html = _composite_html(frames, pattern)

    # APNG with transparency for animation preview + aquarium
    apng_path = _frames_to_apng(frames, max_h=400)
    anim_html = (
        f'<div style="display:flex;justify-content:center;padding:8px 0;">'
        f'<img src="/gradio_api/file={apng_path}" '
        f'style="max-height:260px;object-fit:contain;">'
        f'</div>'
    )

    all_patterns = list(motion_patterns.values()) if motion_patterns else [{}]
    fish_data = {"apng_path": apng_path, "patterns": all_patterns}

    return anim_html, html, fish_data


# ---------------------------------------------------------------------------
#  Step-by-step handlers
# ---------------------------------------------------------------------------

def _render_qr_html(char_id):
    if not char_id or char_id not in CHARS:
        return ""
    return (
        f'<div class="qr-card">'
        f'  <div data-qr-char="{char_id}"></div>'
        f'</div>'
    )


def _render_preview_with_skeleton(char):
    """Render SVG with skeleton keypoints overlay -> PIL Image."""
    svg_path = char["svg"]
    image_rect = compute_image_rect(svg_path)
    _, _, tw, th = image_rect

    img = _render_svg_polygons(svg_path, tw, th)
    for name, (nx, ny) in char["skeleton"]["keypoints"].items():
        x, y = int(nx * tw), int(ny * th)
        cv2.circle(img, (x, y), 5, (0, 0, 255, 255), -1)
        cv2.circle(img, (x, y), 5, (255, 255, 255, 255), 2)
        cv2.putText(img, name, (x + 8, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 0, 255), 1)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))


def _generate_animation_preview(char):
    """Generate animation from original SVG (no photo) -> mp4 path."""
    svg_path = char["svg"]
    image_rect = compute_image_rect(svg_path)
    _, _, tw, th = image_rect
    img_bgra = _render_svg_polygons(svg_path, tw, th)
    keypoints = char["skeleton"]["keypoints"]
    animation_cfg = char["animation"]
    frames = _animate_arap(img_bgra, keypoints, animation_cfg,
                           fps=FPS, duration=DURATION, pad=(40, 60, 40, 40))
    return _frames_to_mp4(frames)


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
        return None, None, None, None, ""
    preview_img = _render_preview_with_skeleton(char)
    anim_path = _generate_animation_preview(char)
    template = char.get("template")
    return preview_img, anim_path, template, template, _render_qr_html(char_id)


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
    mp4_path = _frames_to_mp4(frames)
    return mp4_path, frames


def do_composite(cached_frames, char_id=None):
    if not cached_frames:
        raise gr.Error("Сначала собери анимацию.")
    pattern = None
    if char_id:
        char = CHARS.get(char_id)
        if char:
            motion_patterns = char.get("motion", {})
            if motion_patterns:
                pattern = random.choice(list(motion_patterns.values()))
    html = _composite_html(cached_frames, pattern)
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
