from ultralytics import YOLO

# Load your custom model
model = YOLO(r'C:\Users\umas_local\Documents\user\ws_hema\dell_hema_0026_机械臂\model\data_251215\cucumber_pose_training\exp1\weights\best.pt')

# Perform inference
results = model(r'C:\Users\umas_local\Documents\user\ws_hema\dell_hema_0026_机械臂\model\data_251215\raw_data_251215\2025-12-15-192145.jpg')

# Show results
for r in results:
    print(r.keypoints)  # Print keypoint information
    r.show()  # Display the image with predictions



'''
输出：

   1 conf: tensor([[0.9600, 0.9764]], device='cuda:0')
   - 两个关键点的置信度分别为0.9600和0.9764，非常可靠（>0.9是很好的置信度）

   1 data: tensor([[[702.0493, 257.4019,   0.9600],
   2          [622.0557, 262.6580,   0.9764]]], device='cuda:0')
   - 第一个点(头): x=702.05, y=257.40, 置信度0.9600
   - 第二个点(尾): x=622.06, y=262.66, 置信度0.9764
   - 坐标是像素值

   1 xyn: tensor([[[0.5485, 0.3575],
   2          [0.4860, 0.3648]]], device='cuda:0')
   - 归一化坐标（相对于图像宽高的比例）
   - 第一个点: (0.5485, 0.3575) - 图像右侧
   - 第二个点: (0.4860, 0.3648) - 图像中间偏左
'''