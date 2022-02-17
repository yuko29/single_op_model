# single op model

Generate pytorch models from scratch and the test data

## Procedure
1. Add the python script to generate a python torch model. e.g. `single_conv.py`
    - Store format: {op}/{single|multi}_{op}.onnx 
2. Run the script

## Generate the input and output of the model
1. Edit the `set` section in `setting.ini`, specify 
    - the model
    - the location of the files to be store 
2. Modify `model_test.py` to change the input size if needed
    - the input data would be randomly generated
```
np_array = np.random.randn(1, 1, 32, 32).astype(np.float32)
```
3. Run `model_test.py` to generate `input.pb` and `output.pb`
    -  you will get `{op}_input.pb` and `{op}_output.pb`
