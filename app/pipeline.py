import atexit
import base64
import io
import os
import random
import tempfile
import time
import cv2
import imageio_ffmpeg
import numpy as np
import gradio as gr
from PIL import Image
from rembg import new_session, remove

from svg import SvgCharacter
from config import (
    ARUCO_DICT, MARKER_IDS_BASE, BL_TO_CHAR,
    CANVAS_H, CANVAS_W,
    MARKER_PX, MARGIN, CONTENT_PAD, CONTENT_SCALE,
    FPS, DURATION,
    BACKGROUND_PATH, BACKGROUND_VIDEO_PATH,
    CHARS,
)
from arap import animate_arap as _animate_arap

_SEG_MODEL = "birefnet-general"
_rembg_session_cache = None


def _get_rembg_session():
    """Lazy init + singleton — модель тяжёлая, грузим один раз."""
    global _rembg_session_cache
    if _rembg_session_cache is None:
        _rembg_session_cache = new_session(_SEG_MODEL)
    return _rembg_session_cache

# ---------------------------------------------------------------------------
#  Temp file tracking & cleanup
# ---------------------------------------------------------------------------

_temp_files: list[str] = []


def _track_temp(path: str) -> str:
    _temp_files.append(path)
    return path


def _cleanup_temp():
    for p in _temp_files:
        try:
            os.remove(p)
        except OSError:
            pass
    _temp_files.clear()


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
        known = set(MARKER_IDS_BASE) | set(BL_TO_CHAR.keys())
        return str(len(set(ids.flatten().tolist()) & known))
    except Exception:
        return "0"


# ---------------------------------------------------------------------------
#  Char / background helpers
# ---------------------------------------------------------------------------

def _resolve_char(char_id):
    """Look up character by id, raise gr.Error with friendly message otherwise."""
    char = CHARS.get(char_id)
    if not char:
        raise gr.Error("Персонаж не найден.")
    return char


def _background_markup(full_style="width:100%;height:100%;object-fit:cover;display:block;"):
    """Return HTML for aquarium/composite background (video if available, else image)."""
    if os.path.exists(BACKGROUND_VIDEO_PATH):
        return (
            f'<video autoplay muted loop playsinline style="{full_style}">'
            f'<source src="/gradio_api/file={BACKGROUND_VIDEO_PATH}">'
            '</video>'
        )
    return f'<img src="/gradio_api/file={BACKGROUND_PATH}" style="{full_style}">'


def _pick_motion_pattern(char):
    """Return a random motion pattern dict or None."""
    motion_patterns = char.get("motion", {})
    if not motion_patterns:
        return None
    return random.choice(list(motion_patterns.values()))


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


def align(image, force_char_id=None):
    """Align photo by ArUco markers. Returns (aligned_image, char_id).

    If force_char_id is set — use it (manual mode, для старых шаблонов).
    Иначе — определяется автоматически по BL-маркеру.
    """
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    corners, ids, _ = _aruco_detector.detectMarkers(img)
    if ids is None or len(ids) < 4:
        raise gr.Error(
            f"На фото найдено "
            f"{0 if ids is None else len(ids)} маркеров из 4. "
            "Покажи в кадре все четыре ArUco-маркера."
        )
    detected = {
        int(mid): corners[i][0].mean(axis=0)
        for i, mid in enumerate(ids.flatten())
    }

    bl_ids = set(detected.keys()) - set(MARKER_IDS_BASE)
    if not bl_ids:
        raise gr.Error("Не удалось определить BL-маркер.")
    bl_id = bl_ids.pop()

    if force_char_id:
        if force_char_id not in CHARS:
            raise gr.Error(f"Персонаж {force_char_id} не найден.")
        char_id = force_char_id
    else:
        char_id = BL_TO_CHAR.get(bl_id)
        if not char_id:
            raise gr.Error(f"Неизвестный маркер (ID {bl_id}). Проверь шаблон.")

    marker_ids = MARKER_IDS_BASE + [bl_id]
    src_pts = np.array([detected[mid] for mid in marker_ids], dtype=np.float32)
    dst_pts = np.array(_marker_centers(), dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(
        img, M, (CANVAS_W, CANVAS_H), borderValue=(255, 255, 255)
    )
    return Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)), char_id


# ---------------------------------------------------------------------------
#  Color transfer (by mask)
# ---------------------------------------------------------------------------

def compute_image_rect(svg_path):
    """Compute image_rect from SVG dimensions + template geometry."""
    char = SvgCharacter(svg_path)
    inner = MARGIN + MARKER_PX + CONTENT_PAD
    bx, by = inner, inner
    bw, bh = CANVAS_W - 2 * inner, CANVAS_H - 2 * inner
    scale = min(bw / char.width, bh / char.height) * CONTENT_SCALE
    tw, th = int(char.width * scale), int(char.height * scale)
    return [bx + (bw - tw) // 2, by + (bh - th) // 2, tw, th]


def _correct_photo(aligned_bgr, image_rect):
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
    gain = np.clip(240 / (paper_ref + 1e-6), 0.9, 1.3)

    return np.clip(aligned_bgr.astype(np.float32) * gain[np.newaxis, np.newaxis, :], 0, 255).astype(np.uint8)


def _render_svg_polygons(svg_path, tw, th):
    """Render SVG character (BGRA) for skeleton/animation preview."""
    return SvgCharacter(svg_path).render(tw, th, fill_zones="white")


def _fit_alpha_bbox(src_rgba, dst_rgba):
    """Центрирует и пропорционально масштабирует src_rgba под alpha-bbox dst_rgba.

    Используется чтобы выровнять результат сегментации по SVG-контуру —
    ребёнок мог нарисовать крупнее/смещённо, мы это компенсируем перед анимацией.
    """
    def _bbox(rgba):
        ys, xs = np.where(rgba[..., 3] > 10)
        if len(xs) == 0:
            return None
        return (float(xs.min()), float(ys.min()),
                float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min()))

    sb, db = _bbox(src_rgba), _bbox(dst_rgba)
    if sb is None or db is None:
        return src_rgba

    sx, sy, sw, sh = sb
    dx, dy, dw, dh = db
    scale = min(dw / sw, dh / sh)
    tx = (dx + dw / 2) - (sx + sw / 2) * scale
    ty = (dy + dh / 2) - (sy + sh / 2) * scale

    M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
    h, w = dst_rgba.shape[:2]
    return cv2.warpAffine(src_rgba, M, (w, h),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def digitize_segmentation(aligned_pil, char):
    """Segmentation-based digitize: crop → rembg → fit to SVG bbox. Returns BGRA.

    1. WB-коррекция фото по полям бумаги
    2. rembg отделяет персонажа от фона
    3. fit_alpha_bbox центрирует/масштабирует под SVG — чтобы скелет сел точно
    """
    svg_path = char["svg"]
    image_rect = compute_image_rect(svg_path)
    ix, iy, iw, ih = image_rect

    aligned_bgr = cv2.cvtColor(np.array(aligned_pil), cv2.COLOR_RGB2BGR)
    aligned_bgr = _correct_photo(aligned_bgr, image_rect)
    photo_crop = aligned_bgr[iy : iy + ih, ix : ix + iw]

    _, img_bytes = cv2.imencode(".png", photo_crop)
    fg_png = remove(img_bytes.tobytes(), session=_get_rembg_session())
    fg_raw = cv2.imdecode(np.frombuffer(fg_png, np.uint8), cv2.IMREAD_UNCHANGED)

    svg_contour = SvgCharacter(svg_path).render(iw, ih, fill_zones="transparent")
    return _fit_alpha_bbox(fg_raw, svg_contour)


_ARAP_PAD = (40, 60, 40, 40)  # top, bottom, left, right — запас для выезда при деформации


def _static_frames(rgba, pad=_ARAP_PAD):
    """Для персонажей без анимации — заполняем такой же список кадров,
    но все идентичные. Downstream (mp4/apng/композит) работает без изменений.
    """
    pt, pb, pl, pr = pad
    padded = cv2.copyMakeBorder(rgba, pt, pb, pl, pr,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    return [padded] * (FPS * DURATION)


def _generate_frames(aligned_img, char):
    rgba = digitize_segmentation(aligned_img, char)
    if not char.get("animation_ready"):
        return _static_frames(rgba)
    return _animate_arap(rgba, char["skeleton"]["keypoints"], char["animation"],
                         fps=FPS, duration=DURATION, pad=_ARAP_PAD)


# ---------------------------------------------------------------------------
#  Video encoding
# ---------------------------------------------------------------------------

def _frames_to_mp4(cv_frames):
    """Encode BGRA frames to H.264 mp4 (white background for transparency)."""
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
#  Swim animation (shared by composite + aquarium)
# ---------------------------------------------------------------------------

def _swim_keyframes(swim_cfg, name="swim", offset_y=0):
    """Generate CSS @keyframes + total_dur for one fish path.

    Bakes round-trip path + bob + tilt + turn into a single named @keyframes.
    Used for both single-fish composite and multi-fish aquarium.
    """
    cfg = swim_cfg or {}
    direction = cfg.get("direction", 180)
    flip = cfg.get("flip", False)
    rot = cfg.get("rotate", 0)
    swim_dur = cfg.get("swim_duration", 8)
    bob = cfg.get("bob_range", [40, 55])
    tilt_amp = cfg.get("tilt", 8)
    tilt_dur = cfg.get("tilt_duration", 3) or 3  # guard against 0 / None
    offset_pct = cfg.get("offset_x", 0)

    total_dur = swim_dur * 2  # round trip
    dir_rad = np.radians(direction)
    cos_d, sin_d = np.cos(dir_rad), np.sin(dir_rad)
    span = 35
    ax = 50 - cos_d * span + offset_pct
    ay = 50 + sin_d * span + offset_y
    bx = 50 + cos_d * span + offset_pct
    by = 50 - sin_d * span + offset_y
    bob_amp = (bob[1] - bob[0]) / 2 if isinstance(bob, (list, tuple)) and len(bob) >= 2 else 5
    sx_fwd = -1 if flip else 1
    sx_ret = -sx_fwd
    turn = 0.025  # 2.5% of animation for turn (~0.4s)
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
        if np.isnan(lv) or np.isnan(tv) or np.isnan(r) or np.isnan(sx):
            lv = float(np.nan_to_num(lv, nan=50.0))
            tv = float(np.nan_to_num(tv, nan=50.0))
            r = float(np.nan_to_num(r, nan=0.0))
            sx = float(np.nan_to_num(sx, nan=1.0))
        tf = f"translate(-50%,-50%) rotate({r:.1f}deg) scaleX({sx:.3f})"
        kf_lines.append(
            f"  {t*100:.1f}% {{ left:{lv:.1f}%; top:{tv:.1f}%; transform:{tf}; }}")

    css = f"@keyframes {name} {{\n" + "\n".join(kf_lines) + "\n}"
    return css, total_dur


# ---------------------------------------------------------------------------
#  Composite on background (single fish)
# ---------------------------------------------------------------------------

def _composite_html(cached_frames, swim_cfg=None):
    """Render single-fish composite HTML with background + swim animation."""
    fish_path = _frames_to_apng(cached_frames)
    with open(fish_path, "rb") as f:
        fish_b64 = base64.b64encode(f.read()).decode()

    swim_css, total_dur = _swim_keyframes(swim_cfg or {}, name="swim")

    return f"""\
<div style="position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;border-radius:12px;">
  {_background_markup()}
  <img src="data:image/png;base64,{fish_b64}"
       style="position:absolute;height:35%;
              animation:swim {total_dur}s linear infinite;">
</div>
<style>
{swim_css}
</style>"""


# ---------------------------------------------------------------------------
#  Aquarium (multi-fish)
# ---------------------------------------------------------------------------

AQUARIUM_EMPTY = (
    '<div style="min-height:300px;display:flex;align-items:center;'
    'justify-content:center;color:#9ca3af;border:1px dashed #d1d5db;'
    'border-radius:12px;background:#f8fafc;">'
    '<span style="font-size:18px;">Аквариум пуст — добавь персонажа</span></div>'
)


def _aquarium_html(fish_list):
    """Generate aquarium HTML with inline <style> and unique keyframe names."""
    if not fish_list:
        return AQUARIUM_EMPTY, ""

    # Unique prefix to avoid stale CSS keyframe collisions across updates
    uid = int(time.time() * 1000) % 1_000_000

    bg = _background_markup()
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
    size_slots = [30, 28, 33, 25, 35]
    idx = len(aquarium_state)
    fish = {
        "apng_path": last_fish["apng_path"],
        "swim_cfg": swim_cfg,
        "delay": round(total_dur * (idx * 0.3 % 1), 1),
        "size": size_slots[idx % 5],
        "offset_y": y_slots[idx % 5],
    }
    aquarium_state.append(fish)
    display, btn = _aquarium_html(aquarium_state)
    return aquarium_state, display, btn


def clear_aquarium():
    """Remove all fish from the aquarium."""
    return [], AQUARIUM_EMPTY, ""


# ---------------------------------------------------------------------------
#  Full pipeline (auto mode)
# ---------------------------------------------------------------------------

def auto_process(photo, force_char_id=None):
    """Run align -> animate -> composite. Returns (anim_html, composite_html, fish_data, char_id)."""
    if photo is None:
        return "", "", None, ""

    aligned, char_id = align(photo, force_char_id=force_char_id)
    char = CHARS[char_id]
    frames = _generate_frames(aligned, char)
    html = _composite_html(frames, _pick_motion_pattern(char))

    # APNG with transparency for animation preview + aquarium
    apng_path = _frames_to_apng(frames, max_h=400)
    anim_html = (
        f'<div style="display:flex;justify-content:center;padding:8px 0;">'
        f'<img src="/gradio_api/file={apng_path}" '
        f'style="max-height:260px;object-fit:contain;">'
        f'</div>'
    )

    all_patterns = list(char.get("motion", {}).values()) or [{}]
    fish_data = {"apng_path": apng_path, "patterns": all_patterns}

    return anim_html, html, fish_data, char_id


# ---------------------------------------------------------------------------
#  Step-by-step handlers
# ---------------------------------------------------------------------------

def _render_qr_html(char_id=None):
    """QR card. With char_id → link to ?char=XXX. Without → link to base URL."""
    attr = f'data-qr-char="{char_id}"' if char_id else 'data-qr-char=""'
    return f'<div class="qr-card"><div {attr}></div></div>'


def _render_preview_with_skeleton(char):
    """Render SVG with skeleton keypoints overlay -> PIL Image.

    Для персонажей без skeleton показываем просто SVG без точек.
    """
    svg_path = char["svg"]
    _, _, tw, th = compute_image_rect(svg_path)

    img = _render_svg_polygons(svg_path, tw, th)
    keypoints = char.get("skeleton", {}).get("keypoints", {})
    for name, (nx, ny) in keypoints.items():
        x, y = int(nx * tw), int(ny * th)
        cv2.circle(img, (x, y), 5, (0, 0, 255, 255), -1)
        cv2.circle(img, (x, y), 5, (255, 255, 255, 255), 2)
        cv2.putText(img, name, (x + 8, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 0, 255), 1)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))


def _generate_animation_preview(char):
    """Preview video from original SVG (no photo) -> mp4 path.

    Для персонажей без анимации — статичное видео (SVG-персонаж в покое).
    """
    svg_path = char["svg"]
    _, _, tw, th = compute_image_rect(svg_path)
    img_bgra = _render_svg_polygons(svg_path, tw, th)
    if char.get("animation_ready"):
        frames = _animate_arap(img_bgra, char["skeleton"]["keypoints"], char["animation"],
                               fps=FPS, duration=DURATION, pad=_ARAP_PAD)
    else:
        frames = _static_frames(img_bgra)
    return _frames_to_mp4(frames)


def _on_auto_char_change(char_id):
    char = CHARS.get(char_id) if char_id else None
    template = char.get("template") if char else None
    return template, template


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
    rgba = digitize_segmentation(aligned_img, _resolve_char(char_id))
    return Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))


def do_animation(aligned_img, char_id):
    if aligned_img is None:
        raise gr.Error("Сначала загрузи и выровняй фото рисунка.")
    frames = _generate_frames(aligned_img, _resolve_char(char_id))
    return _frames_to_mp4(frames), frames


def do_composite(cached_frames, char_id=None):
    if not cached_frames:
        raise gr.Error("Сначала собери анимацию.")
    pattern = _pick_motion_pattern(CHARS[char_id]) if char_id and char_id in CHARS else None
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
