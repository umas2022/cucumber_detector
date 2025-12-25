#!/usr/bin/env python3
"""
纯CV蔬菜 + 盒子识别（鲁棒版）
- 支持胡萝卜 / 黄瓜 / 茄子
- 支持暗光
- 支持相机倾斜
- 所有尺寸判断均为比例（分辨率无关）
"""

import cv2
import numpy as np
import os


# =========================
# 全局参数配置
# =========================

VEGETABLE_CLASSES = {
    "cucumber": {
        # "hsv_lower": (30, 10, 5),     # 深绿，放宽 V
        # "hsv_upper": (90, 255, 255),
        # "min_area_ratio": 0.002,
        # "min_length_ratio": 0.18,
        # "min_elongation": 3.0
        "hsv_lower": (30, 20, 20),     # 深绿，放宽 V
        "hsv_upper": (90, 255, 255),
        "min_area_ratio": 0.002,
        "min_length_ratio": 0.01,
        "min_elongation": 3.0
    },
    # "carrot": {
    #     "hsv_lower": (0, 60, 60),
    #     "hsv_upper": (20, 255, 255),
    #     "min_area_ratio": 0.001,
    #     "min_length_ratio": 0.12,
    #     "min_elongation": 2.0
    # },
    # "eggplant": {
    #     "hsv_lower": (120, 20, 20),
    #     "hsv_upper": (160, 255, 255),
    #     "min_area_ratio": 0.0015,
    #     "min_length_ratio": 0.15,
    #     "min_elongation": 2.5
    # }
}

BOX_COLOR = {
    "hsv_lower": (90, 50, 50),   # 蓝色盒子
    "hsv_upper": (130, 255, 255),
    "min_area_ratio": 0.05
}


# =========================
# 工具函数
# =========================

def adjust_gamma(image, gamma=1.3):
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def normalize_point(pt, W, H):
    return (pt[0] / W, pt[1] / H)


def contour_pca_length_width(cnt):
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    pts_c = pts - mean

    cov = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, order]

    proj = pts_c @ eigvecs
    length = proj[:, 0].max() - proj[:, 0].min()
    width = proj[:, 1].max() - proj[:, 1].min()

    direction = eigvecs[:, 0]
    projections = np.dot(pts - mean, direction)
    head = pts[np.argmax(projections)]
    tail = pts[np.argmin(projections)]

    return length, width, tuple(head.astype(int)), tuple(tail.astype(int))


# =========================
# 盒子识别
# =========================

def detect_box(image_bgr):
    H, W = image_bgr.shape[:2]
    img = adjust_gamma(image_bgr, 1.2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array(BOX_COLOR["hsv_lower"]),
        np.array(BOX_COLOR["hsv_upper"])
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) / (H * W) < BOX_COLOR["min_area_ratio"]:
        return None

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)

    center = tuple(np.mean(box, axis=0).astype(int))

    return {
        "corners_px": [tuple(p) for p in box],
        "corners_norm": [normalize_point(p, W, H) for p in box],
        "center_px": center,
        "center_norm": normalize_point(center, W, H)
    }


# =========================
# 蔬菜识别
# =========================

def detect_vegetables(image_bgr):
    H, W = image_bgr.shape[:2]
    img = adjust_gamma(image_bgr, 1.3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    results = []

    for name, cfg in VEGETABLE_CLASSES.items():
        mask = cv2.inRange(
            hsv,
            np.array(cfg["hsv_lower"]),
            np.array(cfg["hsv_upper"])
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area_ratio = cv2.contourArea(cnt) / (H * W)
            if area_ratio < cfg["min_area_ratio"]:
                continue

            length, width, head, tail = contour_pca_length_width(cnt)
            length_ratio = length / max(H, W)
            elongation = length / (width + 1e-6)

            if length_ratio < cfg["min_length_ratio"]:
                continue
            if elongation < cfg["min_elongation"]:
                continue

            center = tuple(np.mean(cnt.reshape(-1, 2), axis=0).astype(int))

            results.append({
                "class": name,
                "head_px": head,
                "tail_px": tail,
                "center_px": center,
                "head_norm": normalize_point(head, W, H),
                "tail_norm": normalize_point(tail, W, H),
                "center_norm": normalize_point(center, W, H),
                "length_ratio": float(length_ratio),
                "elongation": float(elongation)
            })

    return results


# =========================
# 主入口（示例）
# =========================

if __name__ == "__main__":
    img_path = "./img/2025-12-25-105108.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    box = detect_box(img)
    vegetables = detect_vegetables(img)

    debug = img.copy()

    if box:
        for p in box["corners_px"]:
            cv2.circle(debug, p, 8, (255, 0, 0), -1)
        cv2.circle(debug, box["center_px"], 8, (0, 255, 255), -1)

    for v in vegetables:
        cv2.circle(debug, v["head_px"], 7, (0, 0, 255), -1)
        cv2.circle(debug, v["tail_px"], 7, (255, 0, 0), -1)
        cv2.circle(debug, v["center_px"], 6, (0, 255, 0), -1)
        cv2.putText(
            debug,
            v["class"],
            v["center_px"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.imshow("result", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
