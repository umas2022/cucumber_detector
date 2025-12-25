import pyrealsense2 as rs
import numpy as np
import cv2

# =========================
# 1. 创建 pipeline
# =========================

serial = "243622070637"  # ← 你要用的那一台

pipeline = rs.pipeline()
config = rs.config()

config.enable_device(serial)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)


try:
    # =========================
    # 2. 等待一帧数据
    # =========================
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise RuntimeError("未获取到完整帧")

    # =========================
    # 3. 转为 numpy
    # =========================
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 深度单位（米）
    depth_scale = pipeline.get_active_profile() \
                          .get_device() \
                          .first_depth_sensor() \
                          .get_depth_scale()

    print("Depth scale:", depth_scale, "meters/unit")

    # =========================
    # 4. 可视化
    # =========================
    depth_vis = cv2.normalize(
        depth_image, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("RGB", color_image)
    cv2.imshow("Depth", depth_vis)

    cv2.waitKey(0)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
