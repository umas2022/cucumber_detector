from ultralytics import SAM
import cv2
import numpy as np

model = SAM("sam2.1_b.pt")
results = model("2025-12-15-192031.jpg")
result = results[0]

MAX_SIDE = 800   # 最大边长阈值
MIN_SIDE = 20    # 最小边长阈值

filtered_masks = []
filtered_boxes = []

if result.masks is not None:
    masks = result.masks.data.cpu().numpy()      # (N, H, W)
    boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        max_side = max(w, h)
        min_side = min(w, h)

        if min_side < MIN_SIDE:
            continue
        if max_side > MAX_SIDE:
            continue  # 过滤桌面 / 背景

        filtered_masks.append(masks[i])
        filtered_boxes.append(box)

# 手动可视化（只画过滤后的）
img = cv2.imread("2025-12-15-192031.jpg")

for mask in filtered_masks:
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    img[mask.astype(bool)] = img[mask.astype(bool)] * 0.5 + color * 0.5

cv2.imshow("Filtered SAM", img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
