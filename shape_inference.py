import onnx


# Load the ONNX model
model = onnx.load("Alexnet/model.onnx")

inferred_model = onnx.shape_inference.infer_shapes(model)

print(inferred_model.graph.value_info)
