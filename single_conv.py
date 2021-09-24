# Some standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx


class Model(nn.Module): 
    def __init__(self):  # override __init__
        super(Model, self).__init__() # 
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5 ,bias=False) # convolution layer 1
    
    def forward(self, x):
        x = self.conv1(x)
        return x


# Use this an input trace to serialize the model
input_shape = (1, 32, 32)
model_onnx_path = "./Conv/single_conv2d.onnx"
model = Model()
model.train(False)

# Export the model to an ONNX file
input_names = [ "input" ]
output_names = [ "output" ]
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False, input_names=input_names, output_names=output_names,
                          keep_initializers_as_inputs=True)
print("Export of torch_model.onnx complete!")