import onnx
import onnxruntime as ort
import numpy as np
from onnx import numpy_helper
from onnx import version_converter
import configparser

config = configparser.ConfigParser()
config.read('setting.ini')

# Load the ONNX model
model = onnx.load(config['set']['model'])

# Check that the model is well formed
onnx.checker.check_model(model)

# lower the onnx ir_virsion (ONNC available in  ONNX v1.3 / ir_version 3)
target_version = 8
ir_version     = 3

model_path     = config['set']['model']
original_model = onnx.load(model_path)

converted_model = version_converter.convert_version(original_model , target_version)
converted_model.ir_version = ir_version

onnx.save(converted_model, model_path)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

ort_session = ort.InferenceSession(config['set']['model'])

np_array = np.random.randn(1, 1, 32, 32).astype(np.float32)

outputs = ort_session.run(
    None,
    {"input": np_array},
)
# print(outputs[0])

# convert np array to TensorProto
input_tensor = numpy_helper.from_array(np_array)
# save TensorProto
with open(config['set']['save_input'], 'wb') as f:
    f.write(input_tensor.SerializeToString())

output_tensor = np.array(outputs)
output_tensor = numpy_helper.from_array(output_tensor)
with open(config['set']['save_output'], 'wb') as f:
    f.write(output_tensor.SerializeToString())