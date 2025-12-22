import pyrealsense2 as rs

ctx = rs.context()
for dev in ctx.devices:
    print(dev.get_info(rs.camera_info.name),
          dev.get_info(rs.camera_info.serial_number))
