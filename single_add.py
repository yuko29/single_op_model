# Some standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx


class Model(nn.Module): # Model 繼承 nn.Module
    def __init__(self, in_size:int, hidden_size:int, out_size:int):  # override __init__
        super(Model, self).__init__() # 使用父class的__init__()初始化網路
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)
   
    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x
    """
    Combine output with the original input
    """
    def forward(self, x):
        return x + self.convblock(x) # skip connection


# Use this an input trace to serialize the model
input_shape = (1, 32, 32)
model_onnx_path = "./Add/single_add.onnx"
model = Model(in_size=1, hidden_size=6, out_size=5)
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