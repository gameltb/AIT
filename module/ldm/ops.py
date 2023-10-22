from aitemplate.frontend import nn, Tensor

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                device=None, dtype=None) -> None:
        return super().__init__(in_features, out_features, bias, dtype=dtype)

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
    if bias == True and padding_mode == 'zeros':
        return nn.Conv2dBias(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, dtype=dtype)
    else:
        print("error UNIMPLEMENTED")

def time_embed(model_channels, embed_dim, dtype, device):
    return nn.Sequential(
            nn.Linear(model_channels, embed_dim, specialization="swish", dtype=dtype),
            nn.Identity(),
            nn.Linear(embed_dim, embed_dim, dtype=dtype),
        )
GroupNorm = nn.GroupNorm

def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return Conv2d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")
