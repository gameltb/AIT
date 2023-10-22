#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor, DynamicProfileStrategy
from aitemplate.testing import detect_target

from ..util.torch_dtype_from_str import torch_dtype_to_string

import os,sys

# os.environ["AIT_USE_CMAKE_COMPILATION"] = "1"
os.environ["AIT_ENABLE_CUDA_LTO"] = "1"
# sys.setrecursionlimit(10000) #Needed for SDXL


def compile_unet_comfy(
    batch_size=(1, 2),
    height=(8, 128),
    width=(8, 128),
    clip_chunks=(1, 8),
    control=None,
    work_dir="./tmp",
    unet_config={},
    dll_name="UNet2DConditionModel.so",
    model_name="UNet2DConditionModel",
):
    unet_config["dtype"] = torch_dtype_to_string(unet_config["dtype"])
    from ..ldm.modules.diffusionmodules.openaimodel import UNetModel
    unet = UNetModel(**unet_config)
    
    unet.name_parameter_tensor()

    def mark_output(y):
        if type(y) is not tuple:
            y = (y,)
        for i in range(len(y)):
            y[i]._attrs["is_output"] = True
            y[i]._attrs["name"] = "output_%d" % (i)
            y_shape = [d._attrs["values"] for d in y[i]._attrs["shape"]]
            print("AIT output_{} shape: {}".format(i, y_shape))

    batch_size = IntVar(values=batch_size, name="batch_size")
    height = IntVar(values=height, name="height")
    width = IntVar(values=width, name="width")
    prompt_size = IntVar(values=(clip_chunks[0] * 77, clip_chunks[1] * 77))

    hidden_dim = unet_config['context_dim']

    latent_model_input_ait = Tensor([batch_size, height, width, 4], name="x", is_input=True, dtype=unet_config["dtype"])
    timesteps_ait = Tensor([batch_size], name="timesteps", is_input=True, dtype=unet_config["dtype"])
    text_embeddings_pt_ait = Tensor([batch_size, prompt_size, hidden_dim], name="context", is_input=True, dtype=unet_config["dtype"])

    adm_channels = unet_config.get("adm_in_channels", None)
    if adm_channels == None:
        y_ait = None
    else:
        y_ait = Tensor([batch_size, adm_channels], name="y", is_input=True)

    if control == None:
        control_ait = None
    else:
        control_ait = map_control_build_params(control, dtype=unet_config["dtype"])

    Y = unet(
        latent_model_input_ait,
        timesteps=timesteps_ait,
        context=text_embeddings_pt_ait,
        y=y_ait,
        control=control_ait,
    )

    mark_output(Y)

    target = detect_target(
        use_fp16_acc=True, convert_conv_to_gemm=True
    )

    return compile_model(Y, target, work_dir, model_name, dll_name=dll_name)

def map_control_params(control_params):
    root_name = "control"
    ait_control_params = {}
    for key in control_params:
        for i,v in enumerate(control_params[key]):
            ait_control_params[f"{root_name}_{key}_{i}"] = v.half().permute((0, 2, 3, 1)).contiguous()
    return ait_control_params

def map_control_build_params(control_params, dtype):
    root_name = "control"
    ait_control_params = {}
    for key in control_params:
        for i,v in enumerate(control_params[key]):
            if not key in ait_control_params:
                ait_control_params[key] = []
            ait_control_params[key].append(Tensor([v.shape[0],v.shape[2],v.shape[3],v.shape[1]], name=f"{root_name}_{key}_{i}", is_input=True, dtype=dtype))
    return ait_control_params