from dataclasses import dataclass, asdict


@dataclass
class ModuleMetaBase:
    cuda_version: str
    os: str
    sha256: str = ""
    file_size: int = 0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(dict_object):
        return ModuleMetaBase(**dict_object)


def apply_ait_params(aitemplate_module, ait_params: dict):
    aitemplate_module.set_many_constants_with_tensors(ait_params)
    aitemplate_module.fold_constants()
    return aitemplate_module
