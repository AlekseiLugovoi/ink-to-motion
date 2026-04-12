"""ARAP (As-Rigid-As-Possible) деформация и анимация.

Плотный меш + котангентный Лапласиан + итеративный SVD-солвер.
"""

import cv2
import numpy as np


def _warp_triangle(src_img, src_tri, dst_tri, dst_img):
    r = cv2.boundingRect(np.float32([dst_tri]))
    x, y, w, h = r
    x2 = min(x + w, dst_img.shape[1])
    y2 = min(y + h, dst_img.shape[0])
    x, y = max(x, 0), max(y, 0)
    w, h = x2 - x, y2 - y
    if w <= 0 or h <= 0:
        return
    dst_local = np.float32(dst_tri) - np.float32([x, y])
    M = cv2.getAffineTransform(np.float32(src_tri), dst_local)
    warped = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_local), 255)
    roi = dst_img[y:y2, x:x2]
    m = mask[:roi.shape[0], :roi.shape[1]] > 0
    warped = warped[:roi.shape[0], :roi.shape[1]]
    roi[:] = np.where(m[..., None], warped, roi)


def _make_motion(t, base_pts, n_kp, names, motion_cfg):
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


def _build_mesh(keypoints, tw, th, alpha_mask, grid_step=40):
    from scipy.spatial import Delaunay, cKDTree

    kp_names = list(keypoints.keys())
    kp_pts = np.array([[v[0] * tw, v[1] * th] for v in keypoints.values()],
                      dtype=np.float64)
    n_kp = len(kp_pts)

    interior = []
    for y in range(grid_step // 2, th, grid_step):
        for x in range(grid_step // 2, tw, grid_step):
            if alpha_mask[y, x] > 128:
                interior.append([float(x), float(y)])
    interior = np.array(interior, dtype=np.float64) if interior else np.empty((0, 2))

    corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th],
                        [tw/2, 0], [tw, th/2], [tw/2, th], [0, th/2]],
                       dtype=np.float64)

    if len(interior):
        dists, _ = cKDTree(np.vstack([kp_pts, corners])).query(interior)
        interior = interior[dists > grid_step * 0.6]
    n_int = len(interior)

    V0 = np.vstack([kp_pts, interior, corners])
    tri = Delaunay(V0)
    h_idx = list(range(n_kp)) + list(range(n_kp + n_int, len(V0)))

    return V0, tri, n_kp, n_int, kp_names, h_idx


def _precompute(V0, tri, h_idx):
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import factorized

    N = len(V0)
    Wsp = lil_matrix((N, N))
    for s in tri.simplices:
        for k in range(3):
            i, j, o = int(s[k]), int(s[(k+1) % 3]), int(s[(k+2) % 3])
            ei, ej = V0[i] - V0[o], V0[j] - V0[o]
            cross = abs(ei[0] * ej[1] - ei[1] * ej[0])
            cot = np.clip(np.dot(ei, ej) / (cross + 1e-8), 0.01, 100.0)
            Wsp[i, j] += 0.5 * cot
            Wsp[j, i] += 0.5 * cot
    Wsp = Wsp.tocsr()

    L = lil_matrix((N, N))
    for i in range(N):
        row = Wsp[i]
        L[i, i] = row.sum()
        for j in row.indices:
            L[i, int(j)] -= row[0, int(j)]
    for h in h_idx:
        L[h, :] = 0
        L[h, h] = 1.0
    solve_fn = factorized(L.tocsc())

    Wcoo = Wsp.tocoo()
    E_i = Wcoo.row.astype(np.intp)
    E_j = Wcoo.col.astype(np.intp)
    E_w = Wcoo.data.astype(np.float64)
    E_orig = V0[E_i] - V0[E_j]

    return solve_fn, E_i, E_j, E_w, E_orig


def _arap_solve(V0, solve_fn, E_i, E_j, E_w, E_orig, h_idx,
                handle_targets, n_iter=3):
    N = len(V0)
    Vd = V0.copy()
    for k, h in enumerate(h_idx):
        Vd[h] = handle_targets[k]

    for _ in range(n_iter):
        E_def = Vd[E_i] - Vd[E_j]
        outers = E_w[:, None, None] * (E_orig[:, :, None] * E_def[:, None, :])
        S = np.zeros((N, 2, 2))
        np.add.at(S, E_i, outers)

        U, _, Vt = np.linalg.svd(S)
        R = np.einsum('nji,nkj->nik', Vt, U)
        bad = np.linalg.det(R) < 0
        if bad.any():
            U[bad, :, -1] *= -1
            R[bad] = np.einsum('nji,nkj->nik', Vt[bad], U[bad])

        R_sum = R[E_i] + R[E_j]
        contrib = 0.5 * E_w[:, None] * np.einsum('eij,ej->ei', R_sum, E_orig)
        b = np.zeros((N, 2))
        np.add.at(b, E_i, contrib)
        for k, h in enumerate(h_idx):
            b[h] = handle_targets[k]

        Vd[:, 0] = solve_fn(b[:, 0])
        Vd[:, 1] = solve_fn(b[:, 1])

    return Vd


def animate_arap(img_bgra, keypoints, motion_cfg, fps=30, duration=3,
                 pad=(0, 0, 0, 0), grid_step=40, arap_iters=3):
    """Генерирует BGRA-кадры с ARAP-деформацией."""
    if img_bgra.shape[2] == 3:
        img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2BGRA)

    pt, pb, pl, pr = pad
    if any(p > 0 for p in pad):
        oh, ow = img_bgra.shape[:2]
        img_bgra = cv2.copyMakeBorder(
            img_bgra, pt, pb, pl, pr,
            cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        keypoints = {
            name: [(x * ow + pl) / img_bgra.shape[1],
                   (y * oh + pt) / img_bgra.shape[0]]
            for name, (x, y) in keypoints.items()
        }

    th, tw = img_bgra.shape[:2]
    alpha = img_bgra[:, :, 3]

    V0, tri, n_kp, n_int, kp_names, h_idx = _build_mesh(
        keypoints, tw, th, alpha, grid_step)

    corners = V0[n_kp + n_int:]
    kp_pts = V0[:n_kp].copy()
    ctrl_pts = np.vstack([kp_pts, corners])

    solve_fn, E_i, E_j, E_w, E_orig = _precompute(V0, tri, h_idx)

    n_frames = fps * duration
    frames = []
    for fi in range(n_frames):
        t = fi / n_frames
        moved = _make_motion(t, ctrl_pts, n_kp, kp_names, motion_cfg)
        handle_targets = np.vstack([moved[:n_kp], corners])

        Vd = _arap_solve(V0, solve_fn, E_i, E_j, E_w, E_orig,
                         h_idx, handle_targets, arap_iters)

        dst = np.zeros_like(img_bgra)
        for s in tri.simplices:
            _warp_triangle(img_bgra, V0[s].tolist(), Vd[s].tolist(), dst)
        frames.append(dst)

    return frames
