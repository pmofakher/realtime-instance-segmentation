"""
Pre-downloads Mask R-CNN ResNet-50 FPN V2 weights to local PyTorch cache
so that launching the webcam app is instantaneous.
"""
import torch
import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

def download_weights():
    print("[*] Downloading Mask R-CNN ResNet-50 FPN V2 pre-trained weights (COCO)...")
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    _ = maskrcnn_resnet50_fpn_v2(weights=weights)
    print("[+] Model weights cached successfully!")

if __name__ == "__main__":
    download_weights()
