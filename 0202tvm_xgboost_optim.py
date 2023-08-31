import tvm
from tvm import relay
from tvm.relay.backend import Executor
from tvm.relay.backend import Runtime
from tvm.driver import tvmc
from tvm.micro import export_model_library_format

import numpy as np
from utils.utils import generate_model_io_vars_header, extract_io_vars_from_module, _shape_to_size

# Step 1: Load Model
shape_dict = {'input0': [1,1]}
model = tvmc.load('model/sine_mlp3_scripted.pth', shape_dict=shape_dict)
# model = tvmc.load('model/mnist_0.983_quantized.tflite')


RUNTIME = Runtime('crt', {'system-lib':False})
EXECUTOR = Executor('aot',
    {"unpacked-api": True, 
    "interface-api": "c", 
    "workspace-byte-alignment": 4,
    "link-params": True,},
    )

TARGET = tvm.target.target.stm32('stm32F7xx')

with tvm.transform.PassContext(opt_level=3, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": True
                                                    }): # what is usmp? -> Enable Unified Static Memory Planning
    module = relay.build(model.mod, target=TARGET, runtime=RUNTIME, params=model.params, executor=EXECUTOR)


# Step 2: Xgboost Optimize Network



# Step 3: Generate Code Lib

export_model_library_format(module, './models/mnist/default.tar')

input_vars, output_vars = extract_io_vars_from_module(module)
generate_model_io_vars_header(input_vars=input_vars, output_vars=output_vars)