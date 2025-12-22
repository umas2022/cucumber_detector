from ultralytics import SAM
import cv2
import numpy as np

# =========================
# 参数配置
# =========================
IMAGE_PATH = "2025-12-15-192031.jpg"
OUTPUT_PATH = "./result/sam_filtered_output.jpg"

MAX_AREA_RATIO = 0.4     # 过滤大背景（桌面）
MIN_AREA_RATIO = 0.002   # 过滤噪声
EDGE_MARGIN = 30         # 过滤靠近图像边缘的物体（像素）

# =========================
# 1. 加载模型
# =========================
model = SAM("sam2.1_b.pt")

# =========================
# 2. 推理
# =========================
results = model(IMAGE_PATH)
result = results[0]

# =========================
# 3. 读取原图
# =========================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("无法读取输入图片")

H, W = img.shape[:2]
vis_img = img.copy().astype(np.float32)

# =========================
# 4. mask + 边缘联合过滤
# =========================
if result.masks is not None and result.boxes is not None:
    masks = result.masks.data.cpu().numpy()    # (N, H, W)
    boxes = result.boxes.xyxy.cpu().numpy()    # (N, 4)

    for i, (mask, box) in enumerate(zip(masks, boxes)):
        # ---- 面积占比过滤 ----
        area_ratio = mask.sum() / (H * W)
        if area_ratio > MAX_AREA_RATIO:
            continue
        if area_ratio < MIN_AREA_RATIO:
            continue

        # ---- 边缘距离过滤 ----
        x1, y1, x2, y2 = box
        if (
            x1 < EDGE_MARGIN or
            y1 < EDGE_MARGIN or
            (W - x2) < EDGE_MARGIN or
            (H - y2) < EDGE_MARGIN
        ):
            continue

        # ---- 可视化 ----
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        mask_bool = mask.astype(bool)
        vis_img[mask_bool] = (
            vis_img[mask_bool] * 0.5 + color * 0.5
        )

# =========================
# 5. 保存与显示
# =========================
vis_img = vis_img.astype(np.uint8)

cv2.imwrite(OUTPUT_PATH, vis_img)
cv2.imshow("SAM Area + Edge Filtered Result", vis_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
