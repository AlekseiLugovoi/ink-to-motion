import json
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
ASSETS_DIR = APP_DIR / "assets"
TEMPLATES_DIR = (ASSETS_DIR / "templates").resolve()
BACKGROUND_PATH = str((TEMPLATES_DIR / "background.jpg").resolve())
_bg_video_assets = TEMPLATES_DIR / "background.mov"
_bg_video_preproc = REPO_DIR / "preprocessing" / "templates" / "background.mov"
BACKGROUND_VIDEO_PATH = str(_bg_video_assets.resolve()) if _bg_video_assets.exists() else str(_bg_video_preproc.resolve())

# ---------------------------------------------------------------------------
#  Template geometry (A4 @ 150 DPI)
# ---------------------------------------------------------------------------

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]
CANVAS_H, CANVAS_W = 1240, 1754
MARKER_PX, MARGIN, CONTENT_PAD = 80, 60, 10
CONTENT_SCALE = 0.8

# ---------------------------------------------------------------------------
#  Animation
# ---------------------------------------------------------------------------

FPS = 15
DURATION = 2

# ---------------------------------------------------------------------------
#  Auto-discover characters
#  Each char dir must have: mask.svg, skeleton.json, motion.json
# ---------------------------------------------------------------------------

CHARS = {}
for _char_dir in sorted(TEMPLATES_DIR.iterdir()):
    if not _char_dir.is_dir():
        continue
    _char_id = _char_dir.name
    _required = ["mask.svg", "skeleton.json", "motion.json"]
    if all((_char_dir / f).exists() for f in _required):
        with open(_char_dir / "skeleton.json", encoding="utf-8") as _f:
            _skel = json.load(_f)
        with open(_char_dir / "motion.json", encoding="utf-8") as _f:
            _motion = json.load(_f)
        CHARS[_char_id] = {
            "svg": str((_char_dir / "mask.svg").resolve()),
            "skeleton": _skel,
            "motion": _motion,
            "template": str((_char_dir / "template.png").resolve()) if (_char_dir / "template.png").exists() else None,
        }

DEFAULT_CHAR = list(CHARS.keys())[0] if CHARS else None
