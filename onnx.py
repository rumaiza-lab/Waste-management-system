#Python Coding for onnx model
#!/usr/bin/env python3
import torch
import torchvision.models as models
# Define number of classes (as used during training)
num_classes = 2
# Rebuild the MobileNetV2 model architecture
model = models.mobilenet_v2(pretrained=False)
# Replace the classifier head to match our two classes
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
# Load the trained model weights
model_path = "waste_classifier.pth"
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval() # Set model to evaluation mode
# Create a dummy input tensor with the same dimensions as your training images (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 224, 224)
# Define the output ONNX file name
onnx_path = "waste_classifier.onnx"
# Export the model to ONNX format
torch.onnx.export(
model, # model being run
dummy_input, # model input (or a tuple for multiple inputs)
onnx_path, # where to save the model (can be a file or file-like object)
export_params=True, # store the trained parameter weights inside the model file
opset_version=11, # the ONNX version to export the model to
do_constant_folding=True, # whether to execute constant folding for optimization
input_names=["input"], # the model's input names
output_names=["output"], # the model's output names
dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}} # variable length axes
)
print(f"Model successfully exported to {onnx_path}")
