from ultralytics.models.sam import SAM3SemanticPredictor
import numpy as np
import cv2

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,  # Use FP16 for faster inference
    save=False,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image("./img/2025-12-25-105108.jpg")

# Query with multiple text prompts
results = predictor(text=["cucumber"])



# 画图

res = results[0]
img = res.orig_img.copy()

if res.masks is not None:
    masks = res.masks.data.cpu().numpy()  # (N, H, W)

    for mask in masks:
        # 转成 uint8
        mask = (mask > 0.5).astype(np.uint8) * 255

        # 生成彩色 mask（绿色）
        color_mask = np.zeros_like(img)
        color_mask[:, :, 1] = mask  # G 通道

        # 融合
        img = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)

cv2.imshow("SAM3 Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
