import torch
from ultralytics import YOLO
import os


def main():
    # Check if CUDA is available and set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a model - choosing a pose estimation model
    # You can use yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt
    model = YOLO('yolov8n-pose.pt')  # start with a pre-trained model

    # Train the model
    results = model.train(
        data='./cucumber-pose.v2i.yolov8/data.yaml',  # path to your dataset
        epochs=100,                    # number of training epochs
        imgsz=640,                     # input image size
        batch=16,                      # batch size
        device=device,                 # device to run on
        project='cucumber_pose_training',  # project name
        name='exp1',                   # experiment name
        save_period=10,                # save checkpoint every 10 epochs
        optimizer='AdamW',             # optimizer
        lr0=0.001,                     # initial learning rate
        lrf=0.01,                      # final learning rate
        momentum=0.937,                # SGD momentum/Adam beta1
        weight_decay=0.0005,           # optimizer weight decay
        warmup_epochs=3.0,             # warmup epochs
        warmup_momentum=0.8,           # warmup initial momentum
        warmup_bias_lr=0.1,            # warmup initial bias lr
        box=7.5,                       # box loss gain
        cls=0.5,                       # classification loss gain
        dfl=1.5,                       # dfl loss gain
        pose=12.0,                     # pose loss gain
        kobj=2.0,                      # keypoint objectness gain
        label_smoothing=0.0,           # label smoothing
        nbs=64,                        # nominal batch size
        overlap_mask=True,             # masks should overlap during training
        mask_ratio=4,                  # mask downsample ratio
        dropout=0.0,                   # use dropout regularization
        val=True,                      # validate while training
        plots=True,                    # save plots during training
        verbose=False                  # reduce verbosity
    )

    # Validate the model
    print("Validating the model...")
    validation_results = model.val()
    print(f"Validation results: {validation_results}")

    print("Training completed!")
    print("The trained model is saved in cucumber_pose_training/exp/weights/best.pt")


if __name__ == "__main__":
    main()