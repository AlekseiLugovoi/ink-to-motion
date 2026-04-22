"""Camera widget: HTML builder + loads JS/CSS from app/static/."""
import json
from pathlib import Path

from config import MARKER_IDS_BASE, BL_TO_CHAR

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_CAMERA_JS = (_STATIC_DIR / "camera.js").read_text(encoding="utf-8")
CSS = (_STATIC_DIR / "camera.css").read_text(encoding="utf-8")

_KNOWN_ARUCO_IDS = sorted(set(MARKER_IDS_BASE) | set(BL_TO_CHAR.keys()))

CAMERA_HEAD = f"""
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<script>window.__arucoKnownIds = {json.dumps(_KNOWN_ARUCO_IDS)};</script>
<script src="https://cdn.jsdelivr.net/gh/damianofalcioni/js-aruco2@0491d5d228746411d0e9dd602b98f48636a644ba/src/cv.js"></script>
<script src="https://cdn.jsdelivr.net/gh/damianofalcioni/js-aruco2@0491d5d228746411d0e9dd602b98f48636a644ba/src/aruco.js"></script>
<script src="https://cdn.jsdelivr.net/gh/damianofalcioni/js-aruco2@0491d5d228746411d0e9dd602b98f48636a644ba/src/dictionaries/aruco_4x4_1000.js"></script>
<script>
{_CAMERA_JS}
</script>
"""


def make_camera_panel(data_target="camera-data"):
    return f"""
<div class="camera-widget" data-target="{data_target}">
  <button type="button" class="camera-open-btn" data-role="camera-open">Сделать фото</button>
  <div class="camera-status" data-role="camera-status"></div>
  <div class="camera-stage" data-role="camera-stage" hidden>
    <div class="camera-stage-inner">
      <div class="camera-frame">
        <video data-role="camera-video" autoplay playsinline muted></video>
        <div class="camera-frame-overlay">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="17" y="8" width="66" height="84" rx="3" ry="3"
                  fill="none" stroke="rgba(239,68,68,0.95)" stroke-width="0.8"/>
          </svg>
        </div>
      </div>
      <button type="button" class="camera-close-x" data-role="camera-close" disabled>&#10005;</button>
      <div class="camera-badge" data-role="camera-badge">Ищу метки…</div>
      <div class="camera-debug" data-role="camera-debug">init</div>
      <div class="camera-hint">Помести рисунок в рамку и сделай снимок</div>
      <button type="button" class="camera-shutter" data-role="camera-snap" disabled><span></span></button>
    </div>
  </div>
</div>
"""
