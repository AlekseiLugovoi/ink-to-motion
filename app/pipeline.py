import io
import os
import base64
import tempfile
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from scipy.spatial import Delaunay
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
    fill_mask = np.zeros((ih, iw), dtype=np.uint8)

    for pts, fill_bgr, _, _ in ops:
        if fill_bgr == green_bgr:
            cv2.fillPoly(final_alpha, [pts], 255)
            cv2.fillPoly(fill_mask, [pts], 255)
        elif fill_bgr:
            cv2.fillPoly(result, [pts], fill_bgr)
            cv2.fillPoly(final_alpha, [pts], 255)

    mask = fill_mask > 0
    for pts, _, stroke_bgr, thickness in ops:
        if stroke_bgr:
            cv2.polylines(result, [pts], False, stroke_bgr, thickness)
            cv2.polylines(final_alpha, [pts], False, 255, thickness)

    # Закрыть 1px дырки между полигонами (close = dilate + erode)
    final_alpha = cv2.morphologyEx(final_alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

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
    name_to_idx = {n: i for i, n in enumerate(names[:n_kp])}

    for i, name in enumerate(names[:n_kp]):
        if name in motion_cfg:
            m = motion_cfg[name]
            w = a * m.get("freq", 1.0) + m.get("phase", 0) * 2 * np.pi
            pts[i, 0] += m.get("dx", 0) * np.sin(w)
            pts[i, 1] += m.get("dy", 0) * np.sin(w)

    for i, name in enumerate(names[:n_kp]):
        if name in motion_cfg:
            m = motion_cfg[name]
            max_deg = m.get("angle", 0)
            pivot = m.get("pivot")
            if not max_deg or not pivot or pivot not in name_to_idx:
                continue
            pi = name_to_idx[pivot]
            w = a * m.get("freq", 1.0) + m.get("phase", 0) * 2 * np.pi
            bias = m.get("bias", 0.0)
            theta = np.radians(max_deg) * (np.sin(w) + bias)
            ox = base_pts[i, 0] - base_pts[pi, 0]
            oy = base_pts[i, 1] - base_pts[pi, 1]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            pts[i, 0] = pts[pi, 0] + ox * cos_t - oy * sin_t
            pts[i, 1] = pts[pi, 1] + ox * sin_t + oy * cos_t
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


def _frames_to_mp4(cv_frames):
    """Encode BGRA frames to H.264 mp4 (white background for transparency)."""
    import imageio_ffmpeg
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    h, w = cv_frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        tmp.name, (w, h), fps=FPS,
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
        writer.send(rgb.tobytes())
    writer.close()
    return tmp.name


def _frames_to_webm(cv_frames):
    """Encode BGRA frames to VP9 WebM with alpha channel."""
    import imageio_ffmpeg, subprocess
    tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    tmp.close()
    h, w = cv_frames[0].shape[:2]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}",
        "-r", str(FPS), "-i", "pipe:0",
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
        "-b:v", "1M", "-auto-alt-ref", "0",
        tmp.name,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in cv_frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return tmp.name


# ---------------------------------------------------------------------------
#  Composite on background
# ---------------------------------------------------------------------------

def _composite_html(cached_frames):
    fish_path = _frames_to_webm(cached_frames)
    with open(fish_path, "rb") as f:
        fish_b64 = base64.b64encode(f.read()).decode()

    background_markup = ""
    if os.path.exists(BACKGROUND_VIDEO_PATH):
        background_markup = f"""
  <video autoplay muted loop playsinline
         style="width:100%;height:100%;object-fit:cover;display:block;">
    <source src="/gradio_api/file={BACKGROUND_VIDEO_PATH}">
  </video>"""
    else:
        background_markup = f"""
  <img src="/gradio_api/file={BACKGROUND_PATH}"
       style="width:100%;height:100%;object-fit:cover;display:block;">"""

    html = f"""\
<div style="position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;border-radius:12px;cursor:pointer;"
     id="ocean">
  {background_markup}
  <video autoplay muted loop playsinline id="fish"
       src="data:video/webm;base64,{fish_b64}"
       style="position:absolute;height:35%;background:transparent;
              animation:swimH 8s linear infinite, swimV 3s ease-in-out infinite, tilt 3s ease-in-out infinite;">
  </video>
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
    """Run align -> animate -> composite. Returns (mp4_path, composite_html)."""
    if photo is None:
        return None, ""
    char = CHARS.get(char_id)
    if not char:
        raise gr.Error("Персонаж не найден.")

    aligned = align(photo)
    frames = _generate_frames(aligned, char)
    mp4_path = _frames_to_mp4(frames)
    html = _composite_html(frames)
    return mp4_path, html


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

    img_rgba = _render_svg_polygons(svg_path, tw, th)
    keypoints = char["skeleton"]["keypoints"]
    motion_cfg = char["motion"]
    ctrl_pts, tri, n_kp, names = build_triangulation(keypoints, tw, th)

    n_frames = FPS * DURATION
    frames = []
    for i in range(n_frames):
        t = i / n_frames
        dst_pts = make_motion_from_config(t, ctrl_pts, n_kp, names, motion_cfg)
        dst_img = np.zeros_like(img_rgba)
        for simplex in tri.simplices:
            warp_triangle(img_rgba, ctrl_pts[simplex].tolist(),
                          dst_pts[simplex].tolist(), dst_img)
        frames.append(dst_img)
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
