import torch

def torch_dtype_from_str(dtype: str):
    return torch.__dict__.get(dtype, None)

def types_mapping():
    from torch import bfloat16, bool, float16, float32, int32, int64

    yield (float16, "float16")
    yield (bfloat16, "bfloat16")
    yield (float32, "float32")
    yield (int32, "int32")
    yield (int64, "int64")
    yield (bool, "bool")


def torch_dtype_to_string(dtype):
    for (torch_dtype, ait_dtype) in types_mapping():
        if dtype == torch_dtype:
            return ait_dtype
    raise ValueError(
        f"Got unsupported input dtype {dtype}! "
        f"Supported dtypes are: {list(types_mapping())}"
    )


def string_to_torch_dtype(string_dtype):
    if string_dtype is None:
        # Many torch functions take optional dtypes, so
        # handling None is useful here.
        return None

    for (torch_dtype, ait_dtype) in types_mapping():
        if string_dtype == ait_dtype:
            return torch_dtype
    raise ValueError(
        f"Got unsupported ait dtype {string_dtype}! "
        f"Supported dtypes are: {list(types_mapping())}"
    )