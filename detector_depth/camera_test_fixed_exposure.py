import pyrealsense2 as rs
import numpy as np
import cv2

# =========================
# 1. 参数配置（来自你的 AE 结果）
# =========================
SERIAL = "243622070637"   # None 表示不指定
EXPOSURE = 500            # 曝光时间，单位微秒，推荐区间800 ~ 3000
GAIN = 16                # 增益值，同步增大噪声，推荐区间8 ~ 32

# =========================
# 2. 创建 pipeline
# =========================
pipeline = rs.pipeline()
config = rs.config()

if SERIAL is not None:
    config.enable_device(SERIAL)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# =========================
# 3. 配置固定曝光
# =========================
device = profile.get_device()
color_sensor = device.first_color_sensor()

# 关闭自动曝光
color_sensor.set_option(rs.option.enable_auto_exposure, 0)

# 设置固定参数
color_sensor.set_option(rs.option.exposure, EXPOSURE)
color_sensor.set_option(rs.option.gain, GAIN)

print("======== Fixed Exposure Test ========")
print(f"Exposure : {EXPOSURE}")
print(f"Gain     : {GAIN}")
print("====================================")

# =========================
# 4. 连续采集并显示
# =========================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow(
            f"Fixed Exposure (exp={EXPOSURE}, gain={GAIN})",
            color_image
        )

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
