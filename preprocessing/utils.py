import json
import cv2
import numpy as np
from pathlib import Path

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]  # TL, TR, BR, BL


def generate_template(
    img_path=None,
    output_dir="output",
    canvas_size=(1240, 1754),  # A4 landscape @ 150 DPI (h, w)
    marker_px=80,
    margin=60,
    content_pad=10,
    content_scale=1.0,
):
    """Generate printable A4 template with 4 ArUco corner markers.

    If img_path is provided, places the image inside the drawing area.
    content_scale (0..1) controls how much of the drawing area the image fills.
    drawing_bbox is always the full area between markers (independent of content_scale).
    Saves template PNG + meta JSON (marker centers, drawing bbox, canvas size).
    """
    canvas_h, canvas_w = canvas_size
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # --- fixed drawing area between markers ---
    inner = margin + marker_px + content_pad
    drawing_bbox = [inner, inner, canvas_w - 2 * inner, canvas_h - 2 * inner]
    bx, by, bw, bh = drawing_bbox

    # --- place image if provided ---
    image_rect = None
    if img_path is not None:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")

        ih, iw = img.shape[:2]
        scale = min(bw / iw, bh / ih) * content_scale
        tw, th = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

        x0 = bx + (bw - tw) // 2
        y0 = by + (bh - th) // 2
        image_rect = [x0, y0, tw, th]

        if resized.shape[2] == 4:
            alpha = resized[:, :, 3] / 255.0
            for c in range(3):
                canvas[y0 : y0 + th, x0 : x0 + tw, c] = (
                    resized[:, :, c] * alpha
                    + canvas[y0 : y0 + th, x0 : x0 + tw, c] * (1 - alpha)
                ).astype(np.uint8)
        else:
            canvas[y0 : y0 + th, x0 : x0 + tw] = resized

    # --- ArUco markers at corners ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    positions = [
        (margin, margin),
        (canvas_w - margin - marker_px, margin),
        (canvas_w - margin - marker_px, canvas_h - margin - marker_px),
        (margin, canvas_h - margin - marker_px),
    ]

    marker_centers = [
        [px + marker_px / 2, py + marker_px / 2] for px, py in positions
    ]

    for mid, (px, py) in zip(MARKER_IDS, positions):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_px)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[py : py + marker_px, px : px + marker_px] = marker_bgr

    # --- save ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = Path(img_path).stem if img_path else "blank"
    template_path = str(out / f"{stem}_template.png")
    meta_path = str(out / f"{stem}_meta.json")

    cv2.imwrite(template_path, canvas)

    meta = {
        "marker_centers": marker_centers,
        "drawing_bbox": drawing_bbox,
        "image_rect": image_rect,
        "canvas_size": [canvas_h, canvas_w],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "template": canvas,
        "meta": meta,
        "template_path": template_path,
        "meta_path": meta_path,
    }


def detect_and_align(img_path, meta):
    """Detect ArUco markers on photo, warp back to template coordinates."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = detector.detectMarkers(img)

    if ids is None or len(ids) < 4:
        found = 0 if ids is None else len(ids)
        raise ValueError(f"Expected 4 markers, found {found}")

    detected = {}
    for i, mid in enumerate(ids.flatten()):
        detected[int(mid)] = corners[i][0].mean(axis=0)

    src_pts = np.array(
        [detected[mid] for mid in MARKER_IDS], dtype=np.float32
    )
    dst_pts = np.array(meta["marker_centers"], dtype=np.float32)

    canvas_h, canvas_w = meta["canvas_size"]
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(
        img, M, (canvas_w, canvas_h), borderValue=(255, 255, 255)
    )

    return {"aligned": aligned, "matrix": M, "detected_centers": src_pts}


def alignment_error(aligned_img, meta):
    """Re-detect all 16 marker corners on aligned image, compare to expected.

    Homography is built from 4 centers — those always land perfectly.
    The other 12 corner points reveal real distortion (lens, paper warp).
    Returns mean corner error in pixels.
    Guideline: <1px excellent, 1-3px good, >3px consider retaking.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = detector.detectMarkers(aligned_img)

    if ids is None:
        return float("inf"), []

    # expected corner positions from template geometry
    marker_px = meta.get("_marker_px", 80)
    margin = meta.get("_margin", 60)
    canvas_h, canvas_w = meta["canvas_size"]

    positions = [
        (margin, margin),
        (canvas_w - margin - marker_px, margin),
        (canvas_w - margin - marker_px, canvas_h - margin - marker_px),
        (margin, canvas_h - margin - marker_px),
    ]
    # each marker's 4 corners: TL, TR, BR, BL
    expected_corners = {}
    for mid, (px, py) in zip(MARKER_IDS, positions):
        expected_corners[mid] = np.array([
            [px, py], [px + marker_px, py],
            [px + marker_px, py + marker_px], [px, py + marker_px],
        ], dtype=np.float32)

    errors = []
    for i, mid in enumerate(ids.flatten()):
        mid = int(mid)
        if mid in expected_corners:
            detected_4 = corners[i][0]  # shape (4, 2)
            expected_4 = expected_corners[mid]
            for d, e in zip(detected_4, expected_4):
                errors.append(float(np.linalg.norm(d - e)))

    mean_err = np.mean(errors) if errors else float("inf")
    return mean_err, errors


def extract_colored(aligned_img, meta, white_thresh=230):
    """Crop drawing region from aligned image, white -> transparent."""
    x, y, w, h = meta["drawing_bbox"]
    crop = aligned_img[y : y + h, x : x + w]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray < white_thresh).astype(np.uint8) * 255

    b, g, r = cv2.split(crop)
    rgba = cv2.merge([b, g, r, mask])

    return {"rgba": rgba, "crop": crop, "mask": mask}


def transfer_color(aligned_img, original_path, meta, line_thresh=128, sat_thresh=40):
    """Transfer colors from aligned photo onto the original digital drawing.

    Base = original (clean lines). Fill areas (where original is white)
    get color from the aligned photo only where photo has actual color
    (detected via HSV saturation, not brightness — robust to monitor noise).
    """
    if meta.get("image_rect") is None:
        raise ValueError("meta has no image_rect — template was blank")

    ix, iy, iw, ih = meta["image_rect"]

    # crop aligned photo at image_rect
    photo_crop = aligned_img[iy : iy + ih, ix : ix + iw]

    # load and resize original to same size
    original = cv2.imread(str(original_path), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise FileNotFoundError(f"Cannot read: {original_path}")
    original = cv2.resize(original, (iw, ih), interpolation=cv2.INTER_AREA)

    if original.shape[2] == 4:
        original_bgr = original[:, :, :3]
    else:
        original_bgr = original

    # start from original (clean lines on white)
    result = original_bgr.copy()

    # where original is white (fill area) AND photo has actual color → take from photo
    # dilate lines to cover alignment error (few px shift between photo and original)
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    is_line = (original_gray < line_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    is_line_dilated = cv2.dilate(is_line, kernel).astype(bool)
    is_fill = ~is_line_dilated

    photo_hsv = cv2.cvtColor(photo_crop, cv2.COLOR_BGR2HSV)
    is_colored = photo_hsv[:, :, 1] > sat_thresh  # saturation check

    result[is_fill & is_colored] = photo_crop[is_fill & is_colored]

    # alpha: lines + colored fill = visible, white = transparent
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    alpha = (result_gray < 230).astype(np.uint8) * 255

    b, g, r = cv2.split(result)
    rgba = cv2.merge([b, g, r, alpha])

    return {"rgba": rgba, "result": result, "is_fill": is_fill, "is_colored": is_colored}
