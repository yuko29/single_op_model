# Some standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx


class Model(nn.Module): # Model 繼承 nn.Module
    def __init__(self):  # override __init__
        super(Model, self).__init__() # 使用父class的__init__()初始化網路
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
    
    def forward(self, x):
        x = F.max_pool2d(x, (2, 2))
        return x


# Use this an input trace to serialize the model
input_shape = (1, 32, 32)
model_onnx_path = "./MaxPool/single_maxpool.onnx"
model = Model()
#model.train(False)

# test
# input = torch.randn(1, *input_shape)
# print(input)
# output = model(input)
# print(output)

#Export the model to an ONNX file
input_names = [ "input" ]
output_names = [ "output" ]
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False, input_names=input_names, output_names=output_names)
print("Export of torch_model.onnx complete!")