from typing import Union

import json
import os
import torch
import tempfile

from aitemplate.compiler import Model
from . import ModuleMetaBase
from .unnet import ModuleMetaUnet, AITUnetExe
from .vae import ModuleMetaVAE, AITVAEExe

import shutil
import hashlib


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def get_type_name(model_meta):
    return model_meta.__class__.__name__


class AITLoader:
    def __init__(self,
                 modules_path: str = "./modules/",
                 num_runtimes: int = 1,
                 device: Union[str, torch.device] = "cuda",
                 dtype: str = "float16",
                 ) -> None:
        """
        device and dtype can be overriden at the function level
        device must be a cuda device
        """
        self.device = device
        self.dtype = dtype
        self.num_runtimes = num_runtimes
        self.modules_path = modules_path
        try:
            if os.path.exists(f"{modules_path}/modules.json"):
                self.modules = json.load(open(f"{modules_path}/modules.json", "r"))
            else:
                self.modules = {}
        except json.decoder.JSONDecodeError:
            raise ValueError(f"modules.json in {modules_path} is not a valid json file")
        self.temp_dir = tempfile.TemporaryDirectory("ait_build")

    def get_work_dir(self):
        return self.temp_dir.name

    def register_module(self, model_meta: ModuleMetaBase, module_tmp_path):
        type_name = get_type_name(model_meta)
        if not os.path.exists(module_tmp_path):
            return

        sha256 = sha256sum(module_tmp_path)
        module_path = f"{self.modules_path}/{sha256}.so"
        shutil.copy(module_tmp_path, module_path)

        model_meta.sha256 = sha256
        model_meta.file_size = os.path.getsize(module_path)
        if not type_name in self.modules:
            self.modules[type_name] = []
        self.modules[type_name].append(model_meta.to_dict())
        json.dump(self.modules, open(f"{self.modules_path}/modules.json", "w"))

        return model_meta

    def load_module(self, model_meta: ModuleMetaBase):
        module_path = f"{self.modules_path}/{model_meta.sha256}.so"
        return self.load(module_path)

    def load(self, path: str,) -> Model:
        return Model(lib_path=path, num_runtimes=self.num_runtimes)

    def get_cached_module_meta(self,  model_meta: ModuleMetaBase) -> Model:
        type_name = get_type_name(model_meta)
        if type_name in self.modules:
            return self.modules[type_name]
        else:
            return []

    def get_ait_module(self, model_meta: ModuleMetaBase):
        type_map = {
            ModuleMetaUnet: AITUnetExe,
            ModuleMetaVAE: AITVAEExe
        }

        return type_map[type(model_meta)](self, model_meta)
