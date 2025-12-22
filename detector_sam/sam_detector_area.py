from ultralytics import SAM
import cv2
import numpy as np

# =========================
# 参数配置
# =========================
IMAGE_PATH = "2025-12-15-192031.jpg"
OUTPUT_PATH = "./result/sam_filtered_output.jpg"

MAX_AREA_RATIO = 0.4   # mask 占整图面积比例上限，超过认为是背景
MIN_AREA_RATIO = 0.002 # 可选：过滤极小噪声（如碎片）

# =========================
# 1. 加载模型
# =========================
model = SAM("sam2.1_b.pt")

# =========================
# 2. 推理（自动分割）
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
# 4. 按面积占比过滤 mask
# =========================
if result.masks is not None:
    masks = result.masks.data.cpu().numpy()  # (N, H, W)

    for i, mask in enumerate(masks):
        mask_area = mask.sum()
        area_ratio = mask_area / (H * W)

        # 面积过滤条件
        if area_ratio > MAX_AREA_RATIO:
            continue  # 过滤桌面 / 大背景
        if area_ratio < MIN_AREA_RATIO:
            continue  # 过滤噪声

        # 随机颜色
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

        # mask 叠加（半透明）
        mask_bool = mask.astype(bool)
        vis_img[mask_bool] = (
            vis_img[mask_bool] * 0.5 + color * 0.5
        )

# =========================
# 5. 保存与显示结果
# =========================
vis_img = vis_img.astype(np.uint8)

cv2.imwrite(OUTPUT_PATH, vis_img)
cv2.imshow("SAM Area-Filtered Result", vis_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
