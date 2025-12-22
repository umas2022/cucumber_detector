'''
pip install opencv-python numpy

简易cv蔬菜识别，暂不支持多目标
'''

import cv2
import numpy as np
import os

# 预设颜色类别（HSV 范围 + 最小占比/面积）
COLOR_CLASSES = {
    # "green": {  # 黄瓜、青菜
    #     "hsv_lower": (30, 20, 20),
    #     "hsv_upper": (90, 255, 255),
    #     "min_area": 500
    # },
    "red_orange": {  # 番茄、胡萝卜（注意红色在HSV中跨0度，这里简化处理）
        "hsv_lower": (0, 50, 50),
        "hsv_upper": (20, 255, 255),
        "min_area": 500
    },
    # "yellow": {  # 香蕉
    #     "hsv_lower": (15, 80, 80),
    #     "hsv_upper": (35, 255, 255),
    #     "min_area": 500
    # },
    # "purple": {  # 茄子
    #     "hsv_lower": (150, 30, 20),  # 调整色调、饱和度和亮度的下限
    #     "hsv_upper": (270, 255, 255),  # 考虑到色调的周期性，这里设置一个较大的上界
    #     "min_area": 500
    # }
}


def _adjust_gamma(image, gamma=1.3):
    """伽马校正提亮图像"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) *
                     255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def detect_fruit_endpoints(
    image_input,
    gamma=1.3,
    debug=False
):
    """
    检测蔬果图像中的头尾两个端点（支持多颜色类别）

    Args:
        image_input (str or np.ndarray): 图像路径 或 BGR 格式的 numpy 数组
        gamma (float): 伽马校正值，默认 1.3（>1 提亮）
        debug (bool): 是否返回调试信息

    Returns:
        dict: 包含以下字段
            - success (bool)
            - head (tuple or None)
            - tail (tuple or None)
            - class_name (str or None): 识别出的颜色类别（如 'green', 'purple'）
            - debug_info (dict, optional)
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
    img_enhanced = _adjust_gamma(img_bgr, gamma=gamma)
    hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)

    best_result = {"success": False, "head": None,
                   "tail": None, "class_name": None}
    debug_data = {}

    # 2. 尝试每种颜色类别
    for class_name, config in COLOR_CLASSES.items():
        mask = cv2.inRange(hsv, np.array(
            config["hsv_lower"]), np.array(config["hsv_upper"]))

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # 找最大轮廓
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < config["min_area"]:
            continue

        # 拟合主轴并找端点
        try:
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

            # 成功！保存结果并跳出
            best_result = {
                "success": True,
                "head": head,
                "tail": tail,
                "class_name": class_name
            }

            if debug:
                debug_img = original.copy()
                cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(debug_img, head, 8, (0, 0, 255), -1)  # 红色：head
                cv2.circle(debug_img, tail, 8, (255, 0, 0), -1)  # 蓝色：tail
                debug_data = {
                    "mask": mask,
                    "contours": [cnt],
                    "debug_image": debug_img,
                    "class_name": class_name
                }

            break  # 找到第一个有效类别就停止（可改为找面积最大的）

        except Exception as e:
            continue  # 拟合失败则跳过

    result = best_result
    if debug:
        result["debug_info"] = debug_data

    return result


# -------------------------
# 示例用法
# -------------------------
if __name__ == "__main__":
    res = detect_fruit_endpoints("2025-12-19-095730.jpg", debug=True)
    if res["success"]:
        print(f"检测成功！类别: {res['class_name']}")
        print(f"Head: {res['head']}, Tail: {res['tail']}")
        cv2.imshow("Result", res["debug_info"]["debug_image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未能检测到有效蔬果")
