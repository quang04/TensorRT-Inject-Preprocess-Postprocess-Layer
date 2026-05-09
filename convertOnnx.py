import torch
import torchvision.models as models

# python 3.8.10
# torch 2.0.1+cu118
# onnx 1.16.1
# Load pretrained model
model = models.efficientnet_b1(
    weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1
)
model.eval()

# Dummy input fixed 240x240 for efficientNet b1
dummy_input = torch.randn(1, 3, 240, 240)

# Export ONNX widh dynamic batch size
torch.onnx.export(
    model,
    dummy_input,
    "efficientnet_b1.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input" : {
            0: "batch",
            2: "height",
            3: "width"
    }},
)

print("Done")