from urllib.parse import non_hierarchical
import onnx
from onnx.numpy_helper import from_array, to_array
from onnx import helper, TensorProto
from torch import Tensor

# Ref: https://blog.csdn.net/xxradon/article/details/104715524

"""the following code is used to split resnet18-v1-7.onnx"""

"""Load model"""

input_path = 'resnet18-v1-7/resnet18-v1-7.onnx'
output_path = 'resnet18-v1-7/stage4_plus1.onnx'

model = onnx.load(input_path)

"""Extract subgraph"""

oldnodes = [n for n in model.graph.node]
# newnodes = oldnodes[0:10] # stage1_plus0
# newnodes = oldnodes[0:17] # stage1_plus1
# newnodes = oldnodes[0:26] # stage2_plus0
# newnodes = oldnodes[0:33] # stage2_plus1
# newnodes = oldnodes[0:42] # stage3_plus0
# newnodes = oldnodes[0:49] # stage3_plus1
# newnodes = oldnodes[0:58] # stage4_plus0
newnodes = oldnodes[0:65] # stage4_plus1
# newnodes.append(oldnodes[69])
del model.graph.node[:] # clear old nodes
model.graph.node.extend(newnodes)

"""Add new output node"""
# new_output_node = onnx.helper.make_tensor_value_info('resnetv15_stage1__plus0', TensorProto.FLOAT, [1, 64, 56, 56]) # stage1_plus0
# new_output_node = onnx.helper.make_tensor_value_info('resnetv15_stage1__plus1', TensorProto.FLOAT, [1, 64, 56, 56]) # stage1_plus1
# new_output_node = onnx.helper.make_tensor_value_info('resnetv15_stage2__plus0', TensorProto.FLOAT, [1, 128, 28, 28]) # stage2_plus0
# new_output_node = onnx.helper.make_tensor_value_info('resnetv15_stage2__plus1', TensorProto.FLOAT, [1, 128, 28, 28]) # stage2_plus1
# new_output_node = onnx.helper.make_tensor_value_info('resnetv15_stage3__plus0', TensorProto.FLOAT, [1, 256, 14, 14]) # stage3_plus0
# new_output_node = onnx.helper.make_tensor_value_info('resnetv15_stage3__plus1', TensorProto.FLOAT, [1, 256, 14, 14]) # stage3_plus1
# new_output_node = onnx.helper.make_tensor_value_info('resnetv15_stage4__plus0', TensorProto.FLOAT, [1, 512, 7, 7]) # stage4_plus0
new_output_node = onnx.helper.make_tensor_value_info('resnetv15_stage4__plus1', TensorProto.FLOAT, [1, 512, 7, 7]) # stage4_plus1
del model.graph.output[:]
model.graph.output.extend([new_output_node])

"""Save subgraph"""
onnx.save(model, output_path)