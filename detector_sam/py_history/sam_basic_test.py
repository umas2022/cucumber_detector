from ultralytics import SAM
import cv2
import time

img_path = "./img/2025-12-25-105142.jpg"

# =========================
# 1. 统计模型加载时间
# =========================
t0 = time.time()
model = SAM("sam2.1_b.pt")
t1 = time.time()

model_load_time = t1 - t0
print(f"[INFO] Model load time: {model_load_time:.3f} s")

# =========================
# 2. 统计推理时间
# =========================
t2 = time.time()
results = model(img_path)
t3 = time.time()

infer_time = t3 - t2
print(f"[INFO] Inference time: {infer_time:.3f} s")

# =========================
# 3. 后处理 & 可视化（不计时）
# =========================
result = results[0]
plot_img = result.plot()

cv2.imshow("SAM Auto Segmentation", plot_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./result/sam_output.jpg", plot_img)
