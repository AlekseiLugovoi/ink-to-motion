import json
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
TEMPLATES_DIR = (ASSETS_DIR / "templates").resolve()
BACKGROUND_PATH = str((TEMPLATES_DIR / "background.jpg").resolve())
BACKGROUND_VIDEO_PATH = str((TEMPLATES_DIR / "background.mov").resolve())

# ---------------------------------------------------------------------------
#  Template geometry (A4 @ 150 DPI)
# ---------------------------------------------------------------------------

# 47 chars max; for more swap to DICT_4X4_100 (97) or DICT_4X4_250 (247)
ARUCO_DICT = cv2.aruco.DICT_4X4_50
# TL, TR, BR stay fixed; BL encodes character (3 → 001, 4 → 002, 5 → 003, …)
MARKER_IDS_BASE = [0, 1, 2]  # TL, TR, BR — shared by all characters
CANVAS_H, CANVAS_W = 1240, 1754
MARKER_PX, MARGIN, CONTENT_PAD = 80, 60, 10
CONTENT_SCALE = 0.8

# ---------------------------------------------------------------------------
#  Animation
# ---------------------------------------------------------------------------

FPS = 15
DURATION = 3

# ---------------------------------------------------------------------------
#  Auto-discover characters
#  Each char dir must have: mask.svg, skeleton.json, animation.json
# ---------------------------------------------------------------------------

DEFAULT_MOTION = {
    "rotate": 0,
    "swim_duration": 8,
    "bob_range": [40, 55],
    "tilt": 8,
    "tilt_duration": 3,
}


def _parse_motion(raw):
    """Парсит motion.json: один паттерн или dict паттернов.

    Один паттерн: {"rotate": 0, "swim_duration": 8, ...}
    Несколько:    {"swim_left": {...}, "swim_right": {...}}

    Возвращает dict {name: pattern}.
    """
    if not raw:
        return {"default": dict(DEFAULT_MOTION)}
    # Проверяем: если есть вложенные dict — это несколько паттернов
    if any(isinstance(v, dict) for v in raw.values()):
        result = {}
        for name, pat in raw.items():
            merged = dict(DEFAULT_MOTION)
            merged.update(pat)
            result[name] = merged
        return result
    # Один паттерн
    merged = dict(DEFAULT_MOTION)
    merged.update(raw)
    return {"default": merged}

CHARS = {}
_next_bl_id = MARKER_IDS_BASE[-1] + 1  # first BL id = 3
for _char_dir in sorted(TEMPLATES_DIR.iterdir()):
    if not _char_dir.is_dir():
        continue
    _char_id = _char_dir.name
    # Минимум для регистрации — SVG-маска. skeleton/animation опциональны (004 без них).
    if not (_char_dir / "mask.svg").exists():
        continue

    _animation_ready = (
        (_char_dir / "skeleton.json").exists()
        and (_char_dir / "animation.json").exists()
    )
    _skel = {"keypoints": {}}
    _anim = {}
    if _animation_ready:
        with open(_char_dir / "skeleton.json", encoding="utf-8") as _f:
            _skel = json.load(_f)
        with open(_char_dir / "animation.json", encoding="utf-8") as _f:
            _anim = json.load(_f)
    _motion_raw = {}
    if (_char_dir / "motion.json").exists():
        with open(_char_dir / "motion.json", encoding="utf-8") as _f:
            _motion_raw = json.load(_f)
    _motion = _parse_motion(_motion_raw)
    _marker_ids = MARKER_IDS_BASE + [_next_bl_id]
    CHARS[_char_id] = {
        "svg": str((_char_dir / "mask.svg").resolve()),
        "skeleton": _skel,
        "animation": _anim,
        "motion": _motion,
        "template": str((_char_dir / "template.png").resolve()) if (_char_dir / "template.png").exists() else None,
        "marker_ids": _marker_ids,
        "animation_ready": _animation_ready,
    }
    _next_bl_id += 1

DEFAULT_CHAR = list(CHARS.keys())[0] if CHARS else None

# Reverse lookup: BL marker id → char_id
BL_TO_CHAR = {c["marker_ids"][3]: cid for cid, c in CHARS.items()}
