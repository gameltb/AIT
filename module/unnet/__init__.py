from dataclasses import dataclass
from .. import ModuleMetaBase
from . import unet
from .. import apply_ait_params
from ..util import convert_ldm_unet_checkpoint
from ..util.mapping import map_unet

import re


@dataclass
class ModuleMetaUnet(ModuleMetaBase):
    batch_size: list[int, int] = (1, 1)
    width: list[int, int] = (64, 64)
    height: list[int, int] = (64, 64)
    clip_chunks: list[int, int] = (1, 1)
    have_control: bool = False
    unnet_config: object = None

    @staticmethod
    def from_dict(dict_object):
        return ModuleMetaUnet(**dict_object)


class AITUnetExe():
    def __init__(self, ait_loader, model_meta: ModuleMetaUnet) -> None:
        self.model_meta = model_meta
        self.ait_loader = ait_loader
        self.ait_unet_exe = None

    def get_module_cache_key(self):
        key = f"{self.model_meta.os}_{self.model_meta.unnet_config['context_dim']}_{self.model_meta.cuda_version}_{self.model_meta.width}_{self.model_meta.height}_{self.model_meta.batch_size}_{self.model_meta.clip_chunks}"
        key = key.replace(",", "_")
        r = re.compile(r"[() ]")
        key = r.sub("", key)
        return key

    def load_cache_exe(self):
        # find Compatible modules
        cached_module_meta = [ModuleMetaUnet.from_dict(m) for m in self.ait_loader.get_cached_module_meta(self.model_meta)]

        modules = [cache for cache in cached_module_meta
                   if cache.os == self.model_meta.os
                   and cache.cuda_version == self.model_meta.cuda_version
                   and cache.unnet_config['context_dim'] == self.model_meta.unnet_config['context_dim']
                   and cache.batch_size[0] <= self.model_meta.batch_size[0]
                   and cache.batch_size[1] >= self.model_meta.batch_size[1]
                   and cache.width[0] <= self.model_meta.width[0]
                   and cache.width[1] >= self.model_meta.width[1]
                   and cache.height[0] <= self.model_meta.height[0]
                   and cache.height[1] >= self.model_meta.height[1]
                   and cache.clip_chunks[0] <= self.model_meta.clip_chunks[0]
                   and cache.clip_chunks[1] >= self.model_meta.clip_chunks[1]
                   and cache.have_control == self.model_meta.have_control]
        if len(modules) == 0:
            return
        print(f"Found {len(modules)} modules for {self.get_module_cache_key()}")
        print(f"Using {modules[0]}")

        self.ait_unet_exe = self.ait_loader.load_module(modules[0])
        return modules[0]

    def build_exe(self):
        # if detect_target().name() == "rocm":
        #     convert_conv_to_gemm = False
        if self.model_meta.batch_size[1] == 2:
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

            self.model_meta.batch_size = (2, 2)
        else:
            self.model_meta.batch_size = (1, self.model_meta.batch_size[1])

        clip_chunks = self.model_meta.clip_chunks[1]
        if clip_chunks < 10:
            clip_chunks = 10
        self.model_meta.clip_chunks = (1, clip_chunks)

        model_name = self.get_module_cache_key()
        dll_name = model_name + ".dll" if self.model_meta.os == "windows" else model_name + ".so"

        print("building ", self.model_meta)

        self.ait_unet_exe = unet.compile_unet(
            batch_size=self.model_meta.batch_size,
            height=self.model_meta.height,
            width=self.model_meta.width,
            clip_chunks=self.model_meta.clip_chunks,
            convert_conv_to_gemm=True,
            hidden_dim=self.model_meta.unnet_config['context_dim'],
            attention_head_dim=self.model_meta.unnet_config['num_heads'],
            use_linear_projection=self.model_meta.unnet_config['use_linear_in_transformer'],
            block_out_channels=[int(m*self.model_meta.unnet_config['model_channels']) for m in self.model_meta.unnet_config['channel_mult']],
            down_block_types=[
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D"
            ],
            up_block_types=[
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ],
            in_channels=self.model_meta.unnet_config['in_channels'],
            out_channels=self.model_meta.unnet_config['out_channels'],
            class_embed_type=None,
            num_class_embeds=None,
            only_cross_attention=False,
            sample_size=64,
            time_embedding_dim=None,
            conv_in_kernel=3,
            projection_class_embeddings_input_dim=None,
            addition_embed_type=None,
            addition_time_embed_dim=None,
            transformer_layers_per_block=1,
            controlnet=self.model_meta.have_control,
            down_factor=8,
            dtype="float32" if not self.model_meta.unnet_config['use_fp16'] else "float16",
            use_fp16_acc=True if self.model_meta.unnet_config['use_fp16'] else False,
            model_name=model_name,
            work_dir=self.ait_loader.get_work_dir(),
            dll_name=dll_name)

        self.ait_loader.register_module(self.model_meta, f"{self.ait_loader.get_work_dir()}/{model_name}/{dll_name}")

        return self.model_meta

    def apply_ait_params(self, state_dict, device):
        ait_params = map_unet(convert_ldm_unet_checkpoint(state_dict), in_channels=self.model_meta.unnet_config['in_channels'], conv_in_key="conv_in_weight",
                              dim=self.model_meta.unnet_config['model_channels'], device=device, dtype="float32" if not self.model_meta.unnet_config['use_fp16'] else "float16")
        return apply_ait_params(self.ait_unet_exe, ait_params)
