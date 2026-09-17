from torchvision.models import resnet18
import torch

model = resnet18(weights=None)
model.eval()

input = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, input, "resnet18.onnx")
print("Exported model to resnet18.onnx")