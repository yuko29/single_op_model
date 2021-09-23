import onnx
import onnxruntime as ort
import numpy as np
from onnx import numpy_helper



# Load the ONNX model
model = onnx.load("single_conv2d.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

ort_session = ort.InferenceSession("single_maxpool.onnx")

np_array = np.random.randn(1, 1, 32, 32).astype(np.float32)

outputs = ort_session.run(
    None,
    {"input": np_array},
)
# print(outputs[0])

# convert np array to TensorProto
intput_tensor = numpy_helper.from_array(np_array)
# save TensorProto
with open('maxpool_input.pb', 'wb') as f:
    f.write(intput_tensor.SerializeToString())

output_tensor = np.array(outputs)
output_tensor = numpy_helper.from_array(output_tensor)
with open('maxpool_output.pb', 'wb') as f:
    f.write(output_tensor.SerializeToString())