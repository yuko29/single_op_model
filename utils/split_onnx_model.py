import onnx

# the following code is used to split resnet18-v1-7.onnx

def transform(model):
    oldnodes = [n for n in model.graph.node]
    # newnodes = oldnodes[0:10] # 1 block
    newnodes = oldnodes[0:17] # 2 blocks
    # newnodes = oldnodes[0:26] # 3 blocks
    # newnodes = oldnodes[0:33] # 4 blocks
    # newnodes = oldnodes[0:57] # 5 blocks
    # newnodes = oldnodes[0:64] # 6 blocks
    # newnodes = oldnodes[0:73] # 7 blocks
    # newnodes = oldnodes[0:80] # 8 blocks
    del model.graph.node[:] # clear old nodes
    model.graph.node.extend(newnodes)

def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)

input_path = 'resnet18-v1-7/resnet18-v1-7.onnx'
output_path = 'resnet18-v1-7/one_block.onnx'
# input_names = ['data', 'resnetv15_conv0_weight', 'resnetv15_batchnorm0_gamma', 'resnetv15_batchnorm0_beta', 'resnetv15_batchnorm0_running_mean', 'resnetv15_batchnorm0_running_var', 'resnetv15_stage1_conv0_weight', 'resnetv15_stage1_batchnorm0_gamma']
# output_names = ['resnetv15_dense0_fwd', 'resnetv15_conv0_fwd']


apply(transform=transform, infile=input_path, outfile=output_path)

# onnx.utils.extract_model(input_path, output_path, input_names, output_names)