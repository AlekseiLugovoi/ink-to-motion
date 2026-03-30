import os
import io
import json
import base64
import tempfile
import cv2
import numpy as np
import gradio as gr
from PIL import Image
# rembg imported lazily in crop() to avoid slow model download at startup
from scipy.spatial import Delaunay

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]
CANVAS_H, CANVAS_W = 1240, 1754
MARKER_PX, MARGIN, CONTENT_PAD = 80, 60, 10
CONTENT_SCALE = 0.8

TEMPLATES = {
    "Пустой лист": "assets/template.png",
    "Персонаж 001": "assets/img04_template.png",
}

ANIM_CHARS = {
    "Персонаж 001": {
        "drawing": "../preprocessing/templates/001/img.png",
        "mask": "../preprocessing/templates/001/mask.png",
        "skeleton": "../preprocessing/templates/001/skeleton.json",
        "example_gif": "assets/img04_anim.gif",
        "wave_flip": "ltr",
    },
}

FPS = 15
DURATION = 2


# ---------------------------------------------------------------------------
#  Step 1
# ---------------------------------------------------------------------------

def decode_camera_photo(b64_str):
    """Decode base64 data-URL from JS camera capture."""
    if not b64_str:
        return None
    _, data = b64_str.split(",", 1)
    img_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def on_select(name):
    path = TEMPLATES[name]
    return path, path, path


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
    from rembg import remove
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
    """Transfer colors from aligned photo onto original digital character using mask."""
    image_rect = compute_image_rect(char_config["drawing"])
    ix, iy, iw, ih = image_rect

    aligned_bgr = cv2.cvtColor(np.array(aligned_pil), cv2.COLOR_RGB2BGR)
    photo_crop = aligned_bgr[iy : iy + ih, ix : ix + iw]

    mask = cv2.imread(char_config["mask"], cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

    contour = mask < 50
    fill = mask > 200
    between = ~contour & ~fill

    rgba = np.zeros((ih, iw, 4), dtype=np.uint8)
    rgba[contour] = [0, 0, 0, 255]
    rgba[fill, :3] = photo_crop[fill]
    rgba[fill, 3] = 255
    rgba[between] = [0, 0, 0, 0]

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
        raise gr.Error("Для этого шаблона перенос цвета недоступен.")
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


def _rgba_to_gif_frame(frame_rgba):
    """Quantize RGBA frame to palette mode with transparency at index 255."""
    alpha = np.array(frame_rgba.split()[3]) < 128
    frame_p = frame_rgba.convert("RGB").quantize(colors=255)  # индексы 0–254
    palette = frame_p.getpalette()                            # 768 значений
    palette[255 * 3 : 255 * 3 + 3] = [0, 0, 0]              # резервируем индекс 255
    indices = np.array(frame_p, dtype=np.uint8)
    indices[alpha] = 255
    result = Image.fromarray(indices, mode="P")
    result.putpalette(palette)
    result.info['transparency'] = 255
    return result


def _frames_to_gif(pil_frames):
    gif_frames = [_rgba_to_gif_frame(f) for f in pil_frames]
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    gif_frames[0].save(tmp.name, save_all=True, append_images=gif_frames[1:],
                       duration=1000 // FPS, loop=0, disposal=2, transparency=255)
    return tmp.name


def do_animation(aligned_img, selected_name):
    """Step 6 handler — generate animation, cache frames in state."""
    if selected_name not in ANIM_CHARS:
        raise gr.Error("Для этого шаблона анимация недоступна.")
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
    gif_frames = [_rgba_to_gif_frame(f) for f in pil_frames]
    gif_buf = io.BytesIO()
    gif_frames[0].save(gif_buf, format="GIF", save_all=True, append_images=gif_frames[1:],
                       duration=1000 // FPS, loop=0, disposal=2, transparency=255)
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

CAMERA_JS = """
async (dummy) => {
  var isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

  if (isMobile) {
    return await new Promise(function(resolve) {
      var inp = document.createElement('input');
      inp.type = 'file'; inp.accept = 'image/*'; inp.capture = 'environment';
      inp.onchange = function(e) {
        var file = e.target.files[0];
        if (!file) { resolve(null); return; }
        var reader = new FileReader();
        reader.onload = function(ev) { resolve(ev.target.result); };
        reader.readAsDataURL(file);
      };
      inp.click();
    });
  }

  /* Desktop: webcam modal with guide overlay */
  return await new Promise(function(resolve) {
    var corner = 'position:absolute;width:28px;height:28px;border-color:#f97316;border-style:solid;';
    var modal = document.createElement('div');
    modal.id = 'webcam-modal';
    modal.innerHTML =
      '<div style=\"position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;\">' +
        '<div style=\"position:relative;max-width:90%;max-height:70%;\">' +
          '<video id=\"wcam-video\" autoplay playsinline style=\"width:100%;height:100%;border-radius:12px;display:block;\"></video>' +
          '<div style=\"position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;\">' +
            '<div style=\"width:75%;aspect-ratio:1754/1240;max-height:80%;position:relative;box-shadow:0 0 0 9999px rgba(0,0,0,0.45);border-radius:8px;\">' +
              '<div style=\"' + corner + 'top:-2px;left:-2px;border-width:3px 0 0 3px;border-radius:8px 0 0 0;\"></div>' +
              '<div style=\"' + corner + 'top:-2px;right:-2px;border-width:3px 3px 0 0;border-radius:0 8px 0 0;\"></div>' +
              '<div style=\"' + corner + 'bottom:-2px;left:-2px;border-width:0 0 3px 3px;border-radius:0 0 0 8px;\"></div>' +
              '<div style=\"' + corner + 'bottom:-2px;right:-2px;border-width:0 3px 3px 0;border-radius:0 0 8px 0;\"></div>' +
              '<div style=\"position:absolute;bottom:-32px;left:0;right:0;text-align:center;color:rgba(255,200,50,0.9);font-size:14px;text-shadow:0 1px 3px rgba(0,0,0,0.9);\">Совмести лист с рамкой</div>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div style=\"margin-top:16px;display:flex;gap:12px;\">' +
          '<button id=\"wcam-snap\" style=\"padding:12px 32px;font-size:16px;border-radius:8px;border:none;background:#f97316;color:#fff;cursor:pointer;\">Снять</button>' +
          '<button id=\"wcam-close\" style=\"padding:12px 32px;font-size:16px;border-radius:8px;border:none;background:#666;color:#fff;cursor:pointer;\">Отмена</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(modal);

    var stream = null;
    function cleanup() {
      if (stream) stream.getTracks().forEach(function(t){t.stop();});
      modal.remove();
    }

    document.getElementById('wcam-close').onclick = function() { cleanup(); resolve(null); };
    document.getElementById('wcam-snap').onclick = function() {
      var v = document.getElementById('wcam-video');
      var c = document.createElement('canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0);
      var dataUrl = c.toDataURL('image/jpeg', 0.92);
      cleanup();
      resolve(dataUrl);
    };

    navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}}})
      .then(function(s) { stream = s; document.getElementById('wcam-video').srcObject = s; })
      .catch(function(err) { alert('Камера недоступна: ' + err.message); cleanup(); resolve(null); });
  });
}
"""

CAMERA_HEAD = """
<script>
(function() {
  function getTextboxNode(elemId) {
    var root = document.getElementById(elemId);
    if (!root) return null;
    return root.querySelector('textarea, input');
  }

  function setTextboxValue(elemId, value) {
    var node = getTextboxNode(elemId);
    if (!node) return false;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
    if (!setter || !setter.set) {
      setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
    }
    if (!setter || !setter.set) return false;
    setter.set.call(node, value);
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }

  function getTextboxValue(elemId) {
    var node = getTextboxNode(elemId);
    return node ? node.value : '';
  }

  function normalizeTemplateUrl(path) {
    if (!path) return '';
    if (/^(https?:|data:|blob:|\/)/i.test(path)) return path;
    return '/gradio_api/file=' + path.replace(/^\.?\//, '');
  }

  async function getCameraStream(isMobile) {
    var tries = [];
    if (isMobile) {
      tries.push({
        audio: false,
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: { ideal: 'environment' }
        }
      });
    } else {
      try {
        var devices = await navigator.mediaDevices.enumerateDevices();
        var cam = devices.find(function(d) { return d.kind === 'videoinput'; });
        if (cam && cam.deviceId) {
          tries.push({
            audio: false,
            video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              deviceId: { exact: cam.deviceId }
            }
          });
        }
      } catch (err) {
        console.warn('enumerateDevices failed', err);
      }
      tries.push({
        audio: false,
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
    }
    tries.push({ audio: false, video: true });

    var lastErr = null;
    for (var i = 0; i < tries.length; i++) {
      try {
        return await navigator.mediaDevices.getUserMedia(tries[i]);
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr || new Error('Camera stream unavailable');
  }

  function openNativeCameraAndLoad(statusEl) {
    return new Promise(function(resolve) {
      var input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.setAttribute('capture', 'environment');
      input.style.display = 'none';
      document.body.appendChild(input);

      input.onchange = function(e) {
        var file = e.target.files && e.target.files[0];
        if (!file) {
          document.body.removeChild(input);
          resolve(false);
          return;
        }
        var reader = new FileReader();
        reader.onload = function(ev) {
          var ok = setTextboxValue('camera-data', ev.target.result);
          if (!ok && statusEl) statusEl.textContent = 'Native camera opened, but image could not be sent.';
          document.body.removeChild(input);
          resolve(ok);
        };
        reader.onerror = function() {
          if (statusEl) statusEl.textContent = 'Could not read image from native camera.';
          document.body.removeChild(input);
          resolve(false);
        };
        reader.readAsDataURL(file);
      };

      input.click();
    });
  }

  function waitForVideoReady(video, timeoutMs) {
    return new Promise(function(resolve) {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        resolve(true);
        return;
      }

      var done = false;
      var timer = setTimeout(function() {
        if (done) return;
        done = true;
        video.removeEventListener('loadeddata', onReady);
        resolve(false);
      }, timeoutMs || 2500);

      function onReady() {
        if (done) return;
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          done = true;
          clearTimeout(timer);
          video.removeEventListener('loadeddata', onReady);
          resolve(true);
        }
      }

      video.addEventListener('loadeddata', onReady);
    });
  }

  function stopCamera(widget) {
    if (!widget || !widget._cameraStream) return;
    widget._cameraStream.getTracks().forEach(function(track) { track.stop(); });
    widget._cameraStream = null;
  }

  function closeModal(widget) {
    var stage = widget.querySelector('[data-role="camera-stage"]');
    var snapBtn = widget.querySelector('[data-role="camera-snap"]');
    var closeBtn = widget.querySelector('[data-role="camera-close"]');
    stopCamera(widget);
    stage.hidden = true;
    snapBtn.disabled = true;
    closeBtn.disabled = true;
  }

  async function startCamera(widget) {
    var video = widget.querySelector('[data-role="camera-video"]');
    var overlay = widget.querySelector('[data-role="camera-overlay"]');
    var status = widget.querySelector('[data-role="camera-status"]');
    var stage = widget.querySelector('[data-role="camera-stage"]');
    var snapBtn = widget.querySelector('[data-role="camera-snap"]');
    var closeBtn = widget.querySelector('[data-role="camera-close"]');
    var previewImg = document.querySelector('#template-preview img');
    var templatePath = normalizeTemplateUrl(previewImg ? previewImg.src : getTextboxValue('camera-template-src'));
    var isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

    closeModal(widget);
    stage.hidden = false;
    status.textContent = 'Requesting camera access...';

    try {
      var stream = await getCameraStream(isMobile);

      widget._cameraStream = stream;
      video.srcObject = stream;
      overlay.style.display = 'none';
      overlay.onerror = function() { overlay.style.display = 'none'; };
      overlay.onload = function() { overlay.style.display = 'block'; };
      if (templatePath) {
        overlay.src = templatePath;
      } else {
        overlay.removeAttribute('src');
      }
      snapBtn.disabled = false;
      closeBtn.disabled = false;
      status.textContent = 'Align the page with the template and tap Capture.';

      try {
        await video.play();
      } catch (err) {
        console.warn('camera play failed', err);
      }

      var ready = await waitForVideoReady(video, 2500);
      if (!ready) {
        if (isMobile) {
          status.textContent = 'Live camera failed. Opening native camera...';
          closeModal(widget);
          var loaded = await openNativeCameraAndLoad(status);
          if (loaded) {
            status.textContent = 'Photo captured via native camera and loaded.';
          } else {
            status.textContent = 'Live camera failed and native camera was not completed.';
          }
          return;
        }
        status.textContent = 'Camera opened, but video is empty. Check browser camera permission and close other camera apps.';
      }
    } catch (err) {
      console.error(err);
      closeModal(widget);
      status.textContent = 'Camera unavailable. Open via https or localhost and allow browser access.';
    }
  }

  function capturePhoto(widget) {
    var video = widget.querySelector('[data-role="camera-video"]');
    var status = widget.querySelector('[data-role="camera-status"]');

    if (!video.videoWidth || !video.videoHeight) {
      status.textContent = 'Could not capture frame from camera.';
      return;
    }

    var canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    var dataUrl = canvas.toDataURL('image/jpeg', 0.95);

    if (!setTextboxValue('camera-data', dataUrl)) {
      status.textContent = 'Could not send captured image to Gradio.';
      return;
    }

    closeModal(widget);
    status.textContent = 'Photo captured and loaded into the service.';
  }

  function bindCameraWidget() {
    var widget = document.getElementById('camera-widget');
    if (!widget || widget.dataset.bound === '1') return;
    widget.dataset.bound = '1';

    widget.querySelector('[data-role="camera-open"]').addEventListener('click', function() {
      startCamera(widget);
    });

    widget.querySelector('[data-role="camera-snap"]').addEventListener('click', function() {
      capturePhoto(widget);
    });

    widget.querySelector('[data-role="camera-close"]').addEventListener('click', function() {
      closeModal(widget);
      widget.querySelector('[data-role="camera-status"]').textContent = 'Camera is closed.';
    });

    widget.querySelector('[data-role="camera-stage"]').addEventListener('click', function(e) {
      if (e.target.dataset.role === 'camera-stage') {
        closeModal(widget);
      }
    });

    window.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') closeModal(widget);
    });

    window.addEventListener('beforeunload', function() {
      stopCamera(widget);
    });
  }

  window.addEventListener('load', bindCameraWidget);
  new MutationObserver(bindCameraWidget).observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""

CAMERA_PANEL = """
<div id="camera-widget">
  <div class="camera-actions">
    <button type="button" class="camera-btn camera-btn-primary" data-role="camera-open">Open camera</button>
  </div>
  <div class="camera-status" data-role="camera-status">Camera is closed.</div>
  <div class="camera-stage" data-role="camera-stage" hidden>
    <div class="camera-stage-inner">
      <div class="camera-frame">
        <video data-role="camera-video" autoplay playsinline muted></video>
        <img data-role="camera-overlay" alt="template overlay">
        <div class="camera-guide">Align your sheet with the template</div>
      </div>
      <div class="camera-modal-actions">
        <button type="button" class="camera-btn camera-btn-primary camera-btn-lg" data-role="camera-snap" disabled>Capture</button>
        <button type="button" class="camera-btn camera-btn-lg" data-role="camera-close" disabled>Close</button>
      </div>
    </div>
  </div>
</div>
"""

css = """
#main-wrap { max-width: 640px !important; margin: auto !important; }
#camera-data, #camera-template-src { display: none !important; }
#camera-widget { display: grid; gap: 12px; }
#camera-widget .camera-actions { display: flex; gap: 8px; }
#camera-widget .camera-btn {
  border: none;
  border-radius: 12px;
  padding: 10px 14px;
  background: #4b5563;
  color: white;
  cursor: pointer;
  font-size: 14px;
}
#camera-widget .camera-btn-lg {
  padding: 12px 24px;
  font-size: 16px;
  min-width: 120px;
}
#camera-widget .camera-btn[disabled] { opacity: 0.55; cursor: not-allowed; }
#camera-widget .camera-btn-primary { background: #ea580c; }
#camera-widget .camera-status {
  font-size: 13px;
  color: #4b5563;
}
#camera-widget .camera-stage {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(3, 7, 18, 0.86);
  display: none;
  align-items: center;
  justify-content: center;
  padding: 16px;
}
#camera-widget .camera-stage:not([hidden]) {
  display: flex;
}
#camera-widget .camera-stage-inner {
  width: min(900px, 100%);
  display: grid;
  gap: 14px;
}
#camera-widget .camera-frame {
  position: relative;
  overflow: hidden;
  border-radius: 16px;
  background: #111827;
  aspect-ratio: 4 / 3;
}
#camera-widget .camera-frame video,
#camera-widget .camera-frame img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}
#camera-widget .camera-frame video { object-fit: cover; }
#camera-widget .camera-frame img {
  object-fit: contain;
  opacity: 0.32;
  pointer-events: none;
}
#camera-widget .camera-guide {
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 12px;
  padding: 8px 10px;
  border-radius: 10px;
  background: rgba(17, 24, 39, 0.72);
  color: white;
  text-align: center;
  font-size: 13px;
}
#camera-widget .camera-modal-actions {
  display: flex;
  justify-content: center;
  gap: 10px;
}
"""

with gr.Blocks(title="Ink-to-Motion", head=CAMERA_HEAD) as demo:
  with gr.Column(elem_id="main-wrap"):
    gr.Markdown("# Ink-to-Motion")

    with gr.Accordion("Шаг 1. Скачай шаблон", open=True):
        gr.Markdown("Выбери шаблон, распечатай на A4.")
        selector = gr.Dropdown(choices=list(TEMPLATES.keys()), value="Пустой лист", show_label=False)
        preview = gr.Image(value="assets/template.png", show_label=False, interactive=False, height=300, elem_id="template-preview")
        download = gr.DownloadButton("Скачать шаблон", value="assets/template.png", variant="secondary")
        template_src = gr.Textbox(value="assets/template.png", elem_id="camera-template-src", container=False)
        selector.change(fn=on_select, inputs=selector, outputs=[preview, download, template_src])

    with gr.Accordion("Шаг 2. Сфотографируй рисунок", open=False):
        gr.Markdown("Нарисуй персонажа и покажи что получилось.")
        gr.HTML(CAMERA_PANEL)
        camera_data = gr.Textbox(elem_id="camera-data", container=False)
        image_in = gr.Image(show_label=False, type="pil", sources=["upload", "webcam"])
        camera_data.change(fn=decode_camera_photo, inputs=camera_data, outputs=image_in)

    with gr.Accordion("Шаг 3. Выравнивание", open=False):
        align_btn = gr.Button("Выровнять", variant="primary")
        aligned_out = gr.Image(label="Выровненное фото", interactive=False, height=300)
        align_btn.click(fn=align, inputs=image_in, outputs=aligned_out)

    with gr.Accordion("Шаг 4. Вырезка персонажа", open=False):
        crop_btn = gr.Button("Вырезать", variant="primary")
        crop_out = gr.Image(label="Результат", type="pil")
        crop_btn.click(fn=crop, inputs=aligned_out, outputs=crop_out)

    with gr.Accordion("Шаг 5. Перенос цвета + скелет", open=False):
        gr.Markdown("Цвета с фото переносятся на оригинального персонажа.")
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

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
            theme=gr.themes.Soft(primary_hue="orange"), css=css)
