'''
pip install opencv-python numpy
'''

import cv2
import numpy as np
import os

def _adjust_gamma(image, gamma=1.3):
    """伽马校正提亮图像"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def detect_cucumber_endpoints(
    image_input,
    gamma=1.3,
    min_area=500,
    hsv_lower=(30, 20, 20),
    hsv_upper=(90, 255, 255),
    debug=False
):
    """
    检测黄瓜图像中的头尾两个端点（基于 OpenCV）

    Args:
        image_input (str or np.ndarray): 图像路径 或 BGR 格式的 numpy 数组
        gamma (float): 伽马校正值，默认 1.3（>1 提亮）
        min_area (int): 最小轮廓面积阈值（像素），默认 500
        hsv_lower (tuple): HSV 绿色下限，默认 (30, 20, 20)
        hsv_upper (tuple): HSV 绿色上限，默认 (90, 255, 255)
        debug (bool): 是否返回调试信息（掩码、轮廓图等）

    Returns:
        dict: 包含以下字段
            - success (bool): 是否成功检测
            - head (tuple or None): (x, y)，红色端点（投影最大值）
            - tail (tuple or None): (x, y)，蓝色端点（投影最小值）
            - debug_info (dict, optional): 仅当 debug=True 时存在
    """
    # 1. 加载图像
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"图像路径不存在: {image_input}")
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"无法读取图像: {image_input}")
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) != 3 or image_input.shape[2] != 3:
            raise ValueError("输入数组必须是 BGR 格式的 HxWx3 图像")
        img_bgr = image_input.copy()
    else:
        raise TypeError("image_input 必须是文件路径 (str) 或 BGR 图像 (np.ndarray)")

    original = img_bgr.copy()

    # 2. 伽马校正
    img_enhanced = _adjust_gamma(img_bgr, gamma=gamma)

    # 3. HSV 分割
    hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    # 4. 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 5. 轮廓分析
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        result = {"success": False, "head": None, "tail": None}
        if debug:
            result["debug_info"] = {"mask": mask, "contours": []}
        return result

    # 取最大轮廓
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        result = {"success": False, "head": None, "tail": None}
        if debug:
            result["debug_info"] = {"mask": mask, "contours": [cnt]}
        return result

    # 6. 拟合主轴并找端点
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = float(vx), float(vy), float(x), float(y)
    direction = np.array([vx, vy])
    line_point = np.array([x, y])

    cnt_points = cnt.reshape(-1, 2).astype(np.float32)
    projections = np.sum((cnt_points - line_point) * direction, axis=1)
    head_idx = np.argmax(projections)
    tail_idx = np.argmin(projections)

    head = tuple(cnt_points[head_idx].astype(int))
    tail = tuple(cnt_points[tail_idx].astype(int))

    result = {
        "success": True,
        "head": head,
        "tail": tail
    }

    if debug:
        # 绘制调试图
        debug_img = original.copy()
        cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, head, 8, (0, 0, 255), -1)
        cv2.circle(debug_img, tail, 8, (255, 0, 0), -1)
        result["debug_info"] = {
            "mask": mask,
            "contours": [cnt],
            "debug_image": debug_img
        }

    return result


# -------------------------
# 示例用法
# -------------------------
if __name__ == "__main__":
    # 测试单张图
    res = detect_cucumber_endpoints("2025-12-15-193028.jpg", debug=True)
    if res["success"]:
        print(f"Head: {res['head']}, Tail: {res['tail']}")
        cv2.imshow("Debug", res["debug_info"]["debug_image"])
        cv2.waitKey(0)
    else:
        print("检测失败")