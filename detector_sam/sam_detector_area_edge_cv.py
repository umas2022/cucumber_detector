from ultralytics import SAM
import cv2
import numpy as np

# =========================
# 参数配置
# =========================
IMAGE_PATH = "2025-12-15-192031.jpg"
OUTPUT_PATH = "./result/sam_cucumber_result.jpg"

# SAM 后处理
MAX_AREA_RATIO = 0.4
MIN_AREA_RATIO = 0.002
EDGE_MARGIN = 30

# 黄瓜颜色判据
HSV_LOWER = (30, 20, 20)
HSV_UPPER = (90, 255, 255)
CUCUMBER_GREEN_RATIO = 0.25   # mask 内绿色像素占比阈值

# =========================
# 1. 加载模型与图像
# =========================
model = SAM("sam2.1_b.pt")
results = model(IMAGE_PATH)
result = results[0]

img = cv2.imread(IMAGE_PATH)
H, W = img.shape[:2]
vis_img = img.copy().astype(np.float32)

# =========================
# 2. 预处理 HSV（一次即可）
# =========================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
green_mask_full = cv2.inRange(
    hsv,
    np.array(HSV_LOWER),
    np.array(HSV_UPPER)
)

# =========================
# 3. 遍历 SAM mask
# =========================
if result.masks is not None and result.boxes is not None:
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()

    for mask, box in zip(masks, boxes):
        # ---- 面积过滤 ----
        area_ratio = mask.sum() / (H * W)
        if area_ratio > MAX_AREA_RATIO or area_ratio < MIN_AREA_RATIO:
            continue

        # ---- 边缘过滤 ----
        x1, y1, x2, y2 = box
        if (
            x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or
            (W - x2) < EDGE_MARGIN or (H - y2) < EDGE_MARGIN
        ):
            continue

        # ---- mask 内颜色判据 ----
        mask_bool = mask.astype(bool)
        green_pixels = green_mask_full[mask_bool] > 0
        green_ratio = green_pixels.sum() / mask_bool.sum()

        if green_ratio < CUCUMBER_GREEN_RATIO:
            continue  # 不是黄瓜

        # =========================
        # 4. 黄瓜主轴 + 端点计算
        # =========================
        ys, xs = np.where(mask_bool)
        points = np.stack([xs, ys], axis=1).astype(np.float32)

        # PCA 主方向
        mean = points.mean(axis=0)
        centered = points - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]  # 主轴方向

        projections = centered @ direction
        head = points[np.argmax(projections)].astype(int)
        tail = points[np.argmin(projections)].astype(int)

        # =========================
        # 5. 可视化
        # =========================
        color = np.array([0, 255, 0], dtype=np.uint8)
        vis_img[mask_bool] = vis_img[mask_bool] * 0.5 + color * 0.5

        cv2.circle(vis_img, tuple(head), 8, (0, 0, 255), -1)
        cv2.circle(vis_img, tuple(tail), 8, (255, 0, 0), -1)
        cv2.line(vis_img, tuple(head), tuple(tail), (255, 255, 255), 2)

# =========================
# 6. 保存与显示
# =========================
vis_img = vis_img.astype(np.uint8)
cv2.imwrite(OUTPUT_PATH, vis_img)

cv2.imshow("SAM + Color Filtered Cucumber", vis_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
