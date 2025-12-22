from ultralytics import SAM
import cv2

# 1. 加载 SAM 模型
model = SAM("sam2.1_b.pt")

# 2. 推理（全自动模式）
results = model("2025-12-15-192031.jpg")  # 返回 Results 列表

# 3. 获取第一张图的结果（通常只有一张）
result = results[0]

# 4. 可视化：Ultralytics 内置 plot() 方法（推荐！）
plot_img = result.plot()  # 自动将 masks 叠加到原图上，用不同颜色区分物体

# 5. 显示
cv2.imshow("SAM Auto Segmentation", plot_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# （可选）保存结果
cv2.imwrite("./result/sam_output.jpg", plot_img)