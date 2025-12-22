import torch
from ultralytics import YOLO
import os


def main():
    # 检查CUDA可用性
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load a model - 使用中等大小的模型，以平衡精度和速度
    print("Loading YOLOv8 pose model...")
    model = YOLO('../../yolov8n-pose.pt')  # 可以尝试使用 yolov8s-pose.pt 得到更好精度

    # 定义优化的训练参数，针对关键点检测特别优化
    print("Starting training for keypoint detection...")
    results = model.train(
        data='./cucumber-endpoint.v1i.yolov8/data.yaml',  # path to your dataset
        epochs=200,                    # 增加训练轮数以充分收敛
        imgsz=640,                     # 输入图像大小
        batch=8,                       # 减少批大小以获得更精细的梯度更新
        device=device,                 # 使用的设备
        project='cucumber_endpoint_training',  # 项目名称
        name='keypoint_improved',      # 实验名称
        save_period=10,                # 每10轮保存一次权重
        optimizer='AdamW',             # 使用AdamW优化器，对关键点任务更稳定
        lr0=0.0005,                    # 降低初始学习率，提高稳定性 (从0.01降低)
        lrf=0.001,                     # 最终学习率 (从0.01降低)
        momentum=0.08,                 # 调整动量参数
        weight_decay=0.0005,           # 权重衰减
        warmup_epochs=5.0,             # 预热轮数
        warmup_momentum=0.5,           # 预热动量
        warmup_bias_lr=0.1,            # 预热偏置学习率
        
        # 损失函数权重调整，关键点任务需要更精细的权重平衡
        box=7.5,                       # 边界框损失权重
        cls=0.5,                       # 分类损失权重
        dfl=1.5,                       # DFL损失权重
        pose=12.0,                     # 姿态损失权重，针对关键点检测
        kobj=1.0,                      # 关键点对象损失权重
        
        label_smoothing=0.0,           # 标签平滑
        nbs=64,                        # 名义批次大小
        overlap_mask=True,             # 训练时掩码重叠
        mask_ratio=4,                  # 掩码下采样率
        dropout=0.0,                   # 不使用dropout，保持模型能力
        val=True,                      # 训练时验证
        plots=True,                    # 训练时保存图表
        verbose=True,                  # 详细输出
        
        # 数据增强参数调整，对关键点任务更友好的增强
        degrees=5.0,       # 较小的旋转角度以保持关键点准确性
        translate=0.1,      # 平移
        scale=0.5,          # 缩放范围，适度放宽
        shear=2.0,          # 剪切角度
        perspective=0.0,    # 透视变换
        flipud=0.0,         # 不进行上下翻转（保持方向一致性）
        fliplr=0.5,         # 左右翻转概率
        mosaic=0.8,         # 马赛克增强概率，稍微降低
        mixup=0.05,         # MixUp概率，少量使用
        copy_paste=0.0,     # 复制粘贴概率
        
        # 早停和学习率调度
        patience=100,       # 早停轮数，设置较高的值避免过早停止
    )

    # 验证模型
    print("Validating the model...")
    validation_results = model.val()
    print(f"Validation results: {validation_results}")

    print("Training completed!")


if __name__ == "__main__":
    main()