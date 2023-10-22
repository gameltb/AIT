from dataclasses import dataclass
from .. import ModuleMetaBase
from . import unet

import re
import torch

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
        self.ait_exe = None

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

        self.ait_exe = self.ait_loader.load_module(modules[0])
        return modules[0]

    def build_exe(self, control=None):
        # if detect_target().name() == "rocm":
        #     convert_conv_to_gemm = False
        animatediff = False
        if not animatediff:
            if self.model_meta.batch_size[1] <= 2:
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

                self.model_meta.batch_size = (1, 2)
            else:
                self.model_meta.batch_size = (1, self.model_meta.batch_size[1])

            clip_chunks = self.model_meta.clip_chunks[1]
            if clip_chunks < 10:
                clip_chunks = 10
            self.model_meta.clip_chunks = (1, clip_chunks)

        model_name = self.get_module_cache_key()
        dll_name = model_name + ".dll" if self.model_meta.os == "windows" else model_name + ".so"

        print("building ", self.model_meta)

        self.ait_exe = unet.compile_unet_comfy(
            batch_size=self.model_meta.batch_size,
            height=(int(self.model_meta.height[0]/8), int(self.model_meta.height[1]/8)),
            width=(int(self.model_meta.width[0]/8), int(self.model_meta.width[1]/8)),
            clip_chunks=self.model_meta.clip_chunks,
            work_dir=self.ait_loader.get_work_dir(),
            unet_config=self.model_meta.unnet_config, 
            control=control,
            dll_name=dll_name,
            model_name=model_name,
        )

        self.ait_loader.register_module(self.model_meta, f"{self.ait_loader.get_work_dir()}/{model_name}/{dll_name}")

        return self.model_meta

    def set_weights(self, sd):
        constants = map_unet_params(sd)
        self.ait_exe.set_many_constants_with_tensors(constants)

    def apply_model(self, xc, t, context, y=None, control=None, transformer_options=None):
        xc = xc.permute((0, 2, 3, 1)).half().contiguous()
        output = [torch.empty_like(xc)]
        inputs = {"x": xc, "timesteps": t.half(), "context": context.half()}
        if y is not None:
            inputs['y'] = y.half()
        if control is not None:
            control_params = unet.map_control_params(control)
            inputs.update(control_params)
        self.ait_exe.run_with_tensors(inputs, output) #, graph_mode=False)
        return output[0].permute((0, 3, 1, 2))

    def unload_model(self):
        self.ait_exe.close()

def map_unet_params(pt_params):
    dim = 320
    params_ait = {}
    for key, arr in pt_params.items():
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr.half()
    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait