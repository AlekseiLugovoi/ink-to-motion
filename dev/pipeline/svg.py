"""SvgCharacter: единый интерфейс для парсинга и растеризации SVG персонажа.

- Геометрия путей (iter_ops, PaintOp): через svgelements — резолвит CSS, transforms,
  даёт плоский поток PaintOp в painter's order. Используется в pipeline для
  раскраски по фото и выделения зон.
- Растеризация (render): через resvg-py (Rust resvg) — корректно обрабатывает open
  paths по SVG fill-rule. Раньше использовался cv2.fillPoly, но он замыкает
  открытый path прямой линией от end к start, и эта линия может резать тело
  персонажа (см. дельфина 004 — у него обводка тела это один открытый cls-1
  path, и cv2.fillPoly отрезал нижнюю часть брюха/хвоста, делая alpha=0 там).
"""
import re
import cv2
import numpy as np
import resvg_py
from svgelements import SVG, Path as SvgPath, Move, Close

GREEN_BGR = (0, 255, 0)
WHITE_BGR = (255, 255, 255)

# Подменяемые в "white" режиме формы записи зелёного fill в SVG/CSS:
# #00ff00, #0F0, rgb(0,255,0), lime (CSS-keyword, эквивалент #00ff00).
# Зелёный — это служебная разметка зон раскраски (см. process_photo),
# в печатный шаблон она попадать не должна.
_GREEN_PATTERNS = (
    re.compile(r'#0{1,2}[fF]{2}0{1,2}\b'),
    re.compile(r'#0[fF]0\b'),
    re.compile(r'rgb\(\s*0\s*,\s*255\s*,\s*0\s*\)', re.IGNORECASE),
    re.compile(r'\blime\b', re.IGNORECASE),
)
_SVG_OPEN_TAG = re.compile(r'<svg\b[^>]*>', re.IGNORECASE)
_SVG_WIDTH_ATTR = re.compile(r'\s+width\s*=\s*"[^"]*"', re.IGNORECASE)
_SVG_HEIGHT_ATTR = re.compile(r'\s+height\s*=\s*"[^"]*"', re.IGNORECASE)


class PaintOp:
    """Один draw-вызов: полигон точек + цвет заливки/обводки."""
    __slots__ = ("points", "fill_bgr", "stroke_bgr", "thickness")

    def __init__(self, points, fill_bgr, stroke_bgr, thickness=1):
        self.points = points
        self.fill_bgr = fill_bgr
        self.stroke_bgr = stroke_bgr
        self.thickness = thickness


class SvgCharacter:
    """Персонаж из SVG. Скрывает все варианты SVG-разметки."""

    def __init__(self, svg_path):
        self._svg_path = str(svg_path)
        with open(self._svg_path, "r", encoding="utf-8") as f:
            self._svg_text = f.read()
        # reify=True — применяет transforms к координатам и резолвит стили
        self._svg = SVG.parse(self._svg_path, reify=True)

    @property
    def width(self) -> float:
        return float(self._svg.viewbox.width)

    @property
    def height(self) -> float:
        return float(self._svg.viewbox.height)

    def iter_ops(self, sx: float = 1.0, sy: float = 1.0, pts_per_seg: int = 30):
        """Итерация PaintOp в painter's order (как в SVG).

        sx/sy — масштаб координат (например, при рендере на canvas другого размера).
        pts_per_seg — точек на сегмент кривой при семплировании.
        Толщина stroke берётся из SVG (`stroke-width`) и масштабируется.
        """
        scale = (sx + sy) / 2  # средний скейл для толщины stroke (uniform-scaling)
        for el in self._svg.elements():
            if not isinstance(el, SvgPath):
                continue
            fill = _to_bgr(el.fill)
            stroke = _to_bgr(el.stroke)
            if fill is None and stroke is None:
                continue
            sw = float(el.stroke_width or 1) * scale if stroke else 0
            thickness = max(1, int(round(sw)))
            for pts in _sample_subpaths(el, sx, sy, pts_per_seg):
                yield PaintOp(pts, fill, stroke, thickness=thickness)

    def render(self, width: int, height: int, fill_zones: str = "white"):
        """Растеризация персонажа в BGRA массив через resvg.

        fill_zones — что делать с зелёными (#00FF00) fill-зонами:
          - "white": подмена на #ffffff в SVG до рендера (печатные шаблоны)
          - "green": остаются зелёными (для отладки/нотбука)
          - "transparent": зелёные → alpha=0 после рендера (только контур и цветные fill)
        """
        if fill_zones not in {"white", "green", "transparent"}:
            raise ValueError(f"fill_zones must be white/green/transparent, got {fill_zones!r}")

        svg_text = _override_svg_size(self._svg_text, int(width), int(height))
        if fill_zones == "white":
            for pat in _GREEN_PATTERNS:
                svg_text = pat.sub("#ffffff", svg_text)

        png = resvg_py.svg_to_bytes(svg_string=svg_text)
        if isinstance(png, list):  # некоторые версии resvg-py возвращают list[int]
            png = bytes(png)

        rgba = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise RuntimeError("resvg вернул некорректный PNG")
        if rgba.ndim == 2:  # grayscale
            bgr = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
            alpha = np.full(rgba.shape, 255, dtype=np.uint8)
        elif rgba.shape[2] == 3:
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
            alpha = np.full(rgba.shape[:2], 255, dtype=np.uint8)
        else:
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            bgr = bgra[..., :3]
            alpha = bgra[..., 3]
        bgra = cv2.merge([bgr[..., 0], bgr[..., 1], bgr[..., 2], alpha])

        if fill_zones == "transparent":
            g = bgra[..., 1].astype(np.int16)
            r = bgra[..., 2].astype(np.int16)
            b = bgra[..., 0].astype(np.int16)
            green_mask = (g > 200) & (r < 80) & (b < 80)
            bgra[green_mask, 3] = 0

        # resvg при не-аспектном scale может сделать паддинг — гарантируем точный размер.
        if bgra.shape[1] != int(width) or bgra.shape[0] != int(height):
            bgra = cv2.resize(bgra, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        return bgra


def _to_bgr(color):
    """svgelements Color → BGR int-tuple для OpenCV. None если цвет не задан."""
    if color is None or color.value is None:
        return None
    return (int(color.blue), int(color.green), int(color.red))


def _sample_subpaths(path, sx: float, sy: float, n: int):
    """Семплирует path → список np.array(int32) полигонов (по subpath)."""
    subpaths, current = [], []
    for seg in path:
        if isinstance(seg, Move):
            if len(current) >= 3:
                subpaths.append(np.array(current, dtype=np.int32))
            current = [(seg.end.x * sx, seg.end.y * sy)]
        elif isinstance(seg, Close):
            if len(current) >= 3:
                subpaths.append(np.array(current, dtype=np.int32))
            current = []
        else:
            for i in range(1, n + 1):
                pt = seg.point(i / n)
                current.append((pt.x * sx, pt.y * sy))
    if len(current) >= 3:
        subpaths.append(np.array(current, dtype=np.int32))
    return subpaths


def _override_svg_size(svg_text: str, width: int, height: int) -> str:
    """Подменяет width/height на корневом <svg> теге. viewBox остаётся → resvg
    масштабирует содержимое в новые размеры (возможно с искажением аспекта)."""
    def repl(match):
        tag = match.group(0)
        tag = _SVG_WIDTH_ATTR.sub("", tag)
        tag = _SVG_HEIGHT_ATTR.sub("", tag)
        return tag.replace("<svg", f'<svg width="{width}" height="{height}"', 1)
    return _SVG_OPEN_TAG.sub(repl, svg_text, count=1)
