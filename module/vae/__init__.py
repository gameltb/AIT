from dataclasses import dataclass

from .. import ModuleMetaBase
from . import vae
from .. import apply_ait_params
from ..util.mapping import map_vae
from ..util import convert_ldm_vae_checkpoint

import re

@dataclass
class ModuleMetaVAE(ModuleMetaBase):
    batch_size: list[int, int] = (1, 1)
    width: list[int, int] = (64, 64)
    height: list[int, int] = (64, 64)
    vae_type: str = "decode"    
    vae_data_type: str = "float16"
    ddconfig: object = None

    @staticmethod
    def from_dict(dict_object):
        return ModuleMetaVAE(**dict_object)


class AITVAEExe():
    def __init__(self, ait_loader, model_meta: ModuleMetaVAE) -> None:
        self.model_meta = model_meta
        self.ait_loader = ait_loader
        self.ait_exe = None

    def get_module_cache_key(self):
        key = f"{self.model_meta.os}_{self.model_meta.cuda_version}_{self.model_meta.width}_{self.model_meta.height}_{self.model_meta.batch_size}"
        key = key.replace(",", "_")
        r = re.compile(r"[() ]")
        key = r.sub("", key)
        return key

    def load_cache_exe(self):
        # find Compatible modules
        cached_module_meta = [ModuleMetaVAE.from_dict(m) for m in self.ait_loader.get_cached_module_meta(self.model_meta)]

        modules = [cache for cache in cached_module_meta
                   if cache.os == self.model_meta.os
                   and cache.cuda_version == self.model_meta.cuda_version
                   and cache.vae_type == self.model_meta.vae_type
                   and cache.vae_data_type == self.model_meta.vae_data_type
                   and cache.batch_size[0] <= self.model_meta.batch_size[0]
                   and cache.batch_size[1] >= self.model_meta.batch_size[1]
                   and cache.width[0] <= self.model_meta.width[0]
                   and cache.width[1] >= self.model_meta.width[1]
                   and cache.height[0] <= self.model_meta.height[0]
                   and cache.height[1] >= self.model_meta.height[1]]
        if len(modules) == 0:
            return
        print(f"Found {len(modules)} modules for {self.get_module_cache_key()}")
        print(f"Using {modules[0]}")

        self.ait_exe = self.ait_loader.load_module(modules[0])
        return modules[0]

    def build_exe(self):
        # if detect_target().name() == "rocm":
        #     convert_conv_to_gemm = False
        if self.model_meta.batch_size[1] == 1:
            width0 = 64
            width1 = self.model_meta.width[1]
            if self.model_meta.width[1] < 1024:
                width1 = 1024
            self.model_meta.width = (width0, width1)

            height0 = 64
            height1 = self.model_meta.height[1]
            if self.model_meta.height[1] < 1024:
                height1 = 1024
            self.model_meta.height = (height0, height1)

            self.model_meta.batch_size = (1, 1)
        else:
            self.model_meta.batch_size = (1, self.model_meta.batch_size[1])

        model_name = self.get_module_cache_key()
        dll_name = model_name + ".dll" if self.model_meta.os == "windows" else model_name + ".so"

        print("building ", self.model_meta)

        self.ait_exe = vae.compile_vae(
            batch_size=self.model_meta.batch_size,
            height=self.model_meta.height,
            width=self.model_meta.width,
            use_fp16_acc=self.model_meta.vae_data_type == "float16",
            convert_conv_to_gemm=True,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=4,
            in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            sample_size=512,
            input_size=(64, 64),
            down_factor=8,
            dtype=self.model_meta.vae_data_type,
            vae_encode=self.model_meta.vae_type != "decode",
            model_name=model_name,
            work_dir=self.ait_loader.get_work_dir(),
            dll_name=dll_name)

        self.ait_loader.register_module(self.model_meta, f"{self.ait_loader.get_work_dir()}/{model_name}/{dll_name}")

        return self.model_meta

    def apply_ait_params(self, state_dict, device):
        ait_params = map_vae(convert_ldm_vae_checkpoint(state_dict), device=device, dtype=self.model_meta.vae_data_type, encoder=self.model_meta.vae_type != "decode")
        return apply_ait_params(self.ait_exe, ait_params)
