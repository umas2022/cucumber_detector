import pyrealsense2 as rs
import numpy as np
import cv2
import time

# =========================
# 1. 相机序列号（可选）
# =========================
SERIAL = "243622070637"   # 如果只有一台，可以设为 None

# =========================
# 2. 创建 pipeline
# =========================
pipeline = rs.pipeline()
config = rs.config()

if SERIAL is not None:
    config.enable_device(SERIAL)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# =========================
# 3. 获取 color sensor
# =========================
device = profile.get_device()
color_sensor = device.first_color_sensor()

# 开启自动曝光
color_sensor.set_option(rs.option.enable_auto_exposure, 1)

print("Auto Exposure enabled")

# =========================
# 4. 丢弃前 N 帧（关键）
# =========================
WARMUP_FRAMES = 60   # 建议 30~90，约 1~3 秒

for i in range(WARMUP_FRAMES):
    pipeline.wait_for_frames()

print(f"Warmup done ({WARMUP_FRAMES} frames)")

# =========================
# 5. 读取当前曝光参数
# =========================
exposure = color_sensor.get_option(rs.option.exposure)
gain = color_sensor.get_option(rs.option.gain)

print("======== Auto Exposure Result ========")
print(f"Exposure : {exposure:.1f}")
print(f"Gain     : {gain:.1f}")
print("=====================================")

# =========================
# 6. 再取一帧用于观察
# =========================
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

color_image = np.asanyarray(color_frame.get_data())

cv2.imshow("RGB (AE stabilized)", color_image)
cv2.waitKey(0)

pipeline.stop()
cv2.destroyAllWindows()
