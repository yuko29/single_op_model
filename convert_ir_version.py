import onnx
import onnxruntime as ort
import numpy as np
from onnx import numpy_helper
from onnx import version_converter
import configparser

# Workaround for version_converter report error like: Input conv_first.weight is undefined!
# Including all the initializers in the graph inputs can be the solution
# Provided by https://github.com/onnx/onnx/issues/2995#issuecomment-687631931
def add_input_from_initializer(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.input}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.input.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    return add_const_value_infos_to_graph(model.graph)

config = configparser.ConfigParser()
config.read('setting.ini')

# Load the ONNX model
model = onnx.load(config['set']['model'])

# Check that the model is well formed
onnx.checker.check_model(model)

# lower the onnx ir_virsion (ONNC available in  ONNX v1.3 / ir_version 3)
target_op_version = 8   # compatible opset version
ir_version     = 3

model_path     = config['set']['model']
original_model = onnx.load(model_path)
print("original ir_version: ", original_model.ir_version)

add_input_from_initializer(original_model)

converted_model = version_converter.convert_version(original_model , target_version=target_op_version)
converted_model.ir_version = ir_version
print("converted ir_version: ", converted_model.ir_version)

# save converted model
onnx.save(converted_model, model_path)