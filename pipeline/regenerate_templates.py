"""One-shot: regenerate all character templates with unique BL marker IDs."""
import shutil
import tempfile
from pathlib import Path

from utils import generate_template, overlay_character

PIPELINE_DIR = Path(__file__).resolve().parent
APP_ASSETS_DIR = PIPELINE_DIR.parent / "app" / "assets" / "templates"

CANVAS_SIZE = (1240, 1754)
MARKER_PX = 80
MARGIN = 60
CONTENT_PAD = 10
TITLE = "SKETCH FACTORY"

BASE_MARKERS = [0, 1, 2]
BL_START = 3  # 001 → 3, 002 → 4, 003 → 5, ...

char_dirs = sorted(p for p in (PIPELINE_DIR / "templates").iterdir()
                   if p.is_dir() and (p / "mask.svg").exists())

for idx, char_dir in enumerate(char_dirs):
    char_id = char_dir.name
    bl_id = BL_START + idx
    marker_ids = BASE_MARKERS + [bl_id]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        blank_path = tmp.name

    try:
        generate_template(
            output_path=blank_path,
            canvas_size=CANVAS_SIZE,
            marker_ids=marker_ids,
            marker_px=MARKER_PX,
            margin=MARGIN,
            content_pad=CONTENT_PAD,
            title=TITLE,
        )
        out = char_dir / "template.png"
        overlay_character(
            template_path=blank_path,
            svg_path=str(char_dir / "mask.svg"),
            output_path=str(out),
            content_scale=0.8,
            canvas_size=CANVAS_SIZE,
            marker_px=MARKER_PX,
            margin=MARGIN,
            content_pad=CONTENT_PAD,
        )
        # mirror to app/assets
        app_target = APP_ASSETS_DIR / char_id / "template.png"
        app_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, app_target)
        print(f"{char_id}: marker_ids={marker_ids} -> {out}")
    finally:
        Path(blank_path).unlink(missing_ok=True)
