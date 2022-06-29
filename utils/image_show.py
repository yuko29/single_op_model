from unittest import TestCase
import onnx
import numpy as np
from onnx import numpy_helper
from onnx import version_converter
import os
import torch 
import matplotlib.pyplot as plt

test_case = onnx.TensorProto()
# with open(os.path.join('Alexnet/test_data_set_0', 'input_0.pb'), 'rb') as f:
with open(os.path.join('resnet18-v1-7/test_data_set_0', 'input_0.pb'), 'rb') as f:
    test_case.ParseFromString(f.read())

# re_img = np.frombuffer(test_case, dtype=np.uint8)
test_case = numpy_helper.to_array(test_case)

# transpose (NCHW -> NHWC)
test_case = torch.from_numpy(test_case)
test_case = test_case.permute(0, 2, 3, 1)
print(test_case.shape)


plt.imshow(test_case[0])
# plt.pause(0.5)
plt.show()