import onnx


# Load the ONNX model
model = onnx.load("resnet18-v1-7/stage4_plus1.onnx")

inferred_model = onnx.shape_inference.infer_shapes(model)

print(inferred_model.graph.value_info)
