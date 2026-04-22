"""Параметры dev-пайплайна (sandbox). app/ не затрагивает."""
from pathlib import Path

# --- геометрия шаблона (A4 @ 150 DPI) ---
# A4 = 210×297мм. При 150 DPI: 1240×1754 px — компромисс между качеством
# и скоростью ArUco/warp/color-transfer. 300 DPI было бы в 4× медленнее без
# заметного выигрыша для детских раскрасок.
CANVAS_SIZE = (1240, 1754)    # (h, w)
CANVAS_H, CANVAS_W = CANVAS_SIZE

MARKER_PX = 80                # ≈13.5мм на распечатке — надёжная детекция с телефона
MARGIN = 60                   # ≈10мм — поле принтера (иначе маркеры обрежутся)
CONTENT_PAD = 10              # ≈1.7мм — зазор между маркером и зоной рисунка
CONTENT_SCALE = 0.8           # рисунок = 80% зоны (запас на ARAP-деформацию)

TITLE = "SKETCH FACTORY"      # текст в шапке шаблона

# --- производные значения (считаются один раз при импорте) ---
_INNER = MARGIN + MARKER_PX + CONTENT_PAD
DRAWING_BBOX = (_INNER, _INNER, CANVAS_W - 2 * _INNER, CANVAS_H - 2 * _INNER)
MARKER_CENTERS = [
    (MARGIN + MARKER_PX / 2, MARGIN + MARKER_PX / 2),                          # TL
    (CANVAS_W - MARGIN - MARKER_PX / 2, MARGIN + MARKER_PX / 2),               # TR
    (CANVAS_W - MARGIN - MARKER_PX / 2, CANVAS_H - MARGIN - MARKER_PX / 2),    # BR
    (MARGIN + MARKER_PX / 2, CANVAS_H - MARGIN - MARKER_PX / 2),               # BL
]

# --- анимация (превью в ноутбуке) ---
FPS = 30
DURATION = 3                  # секунды

# --- пути ---
PIPELINE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PIPELINE_DIR / "templates"
INPUT_DIR = PIPELINE_DIR / "input"
OUTPUT_DIR = PIPELINE_DIR / "output"
