import torch
from torchvision.models import ResNet18_Weights, resnet18


def main():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    sample_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        sample_input,
        "resnet18.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    print("Exported pretrained ResNet18 to resnet18.onnx")


if __name__ == "__main__":
    main()