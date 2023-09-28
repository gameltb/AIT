import os
import torch
import math
import copy
import itertools

from module.loader import AITLoader
from module.unnet import ModuleMetaUnet
from module.vae import ModuleMetaVAE

base_path = os.path.dirname(os.path.realpath(__file__))
modules_dir = os.path.join(base_path, "modules_cache")
current_loaded_model = None

modules_path = str(modules_dir).replace("\\", "/")
AITLOADER = AITLoader(modules_path)
AIT_OS = "windows" if os.name == "nt" else "linux"
cuda = torch.cuda.get_device_capability()
if cuda[0] == 7 and cuda[1] == 5:
    AIT_CUDA = "sm75"
elif cuda[0] == 7 and cuda[1] == 0:
    AIT_CUDA = "sm70"
elif cuda[0] >= 8:
    AIT_CUDA = "sm80"
else:
    raise ValueError(f"Unsupported CUDA version {cuda[0]}.{cuda[1]}")

# SD15 = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
#         'adm_in_channels': None, 'use_fp16': True, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': 2,
#         'attention_resolutions': [1, 2, 4], 'transformer_depth': [1, 1, 1, 0], 'channel_mult': [1, 2, 4, 4],
#         'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, "num_heads": 8}
# module_meta = ModuleMetaUnet(os=AIT_OS, cuda_version=AIT_CUDA, batch_size=(1, 1),
#                              width=(64, 512), height=(512, 512), clip_chunks=(1, 1), unnet_config=SD15)

# module = AITLOADER.get_ait_module(module_meta)

# m = module.load_cache_exe()
# print(m)
# if m ==None:
#     m = module.build_exe()
#     print(m)


module_meta = ModuleMetaVAE(os=AIT_OS, cuda_version=AIT_CUDA, batch_size=(1, 1),
                             width=(64, 512), height=(512, 512))

module = AITLOADER.get_ait_module(module_meta)

m = module.load_cache_exe()
print(m)
if m ==None:
    m = module.build_exe()
    print(m)
# for sd_version, width, height, model_type, clip_chunks in itertools.product(["v1"], [1024], [1024], ["unet", "vae_decode", "unet_control", "controlnet"], [10]):
#     module_meta = ModuleMeta(os=AIT_OS, sd_version="v1", cuda_version=AIT_CUDA, batch_size=1,
#                             width=width, height=height, model_type=model_type, clip_chunks=clip_chunks)
#     # load the module
#     module, is_cache = AITemplateManger.load_by_model_meta(module_meta)
#     AITemplateManger.unload()

# for sd_version, width, height, model_type, clip_chunks in itertools.product(["v1"], [512], [768], ["unet", "unet_control", "controlnet"], [10]):
#     module_meta = ModuleMeta(os=AIT_OS, sd_version="v1", cuda_version=AIT_CUDA, batch_size=6,
#                             width=width, height=height, model_type=model_type, clip_chunks=clip_chunks)
#     # load the module
#     module, is_cache = AITemplateManger.load_by_model_meta(module_meta)
#     AITemplateManger.unload()
