import os
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.sd

import torch
import copy

# from .module.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from .module.util.torch_dtype_from_str import torch_dtype_to_string
from .module.loader import AITLoader
from .module.unnet import ModuleMetaUnet
from .module.inference import controlnet_inference, vae_inference
from .module.vae import ModuleMetaVAE

MAX_RESOLUTION = 8192

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


class ControlNet(comfy.controlnet.ControlNet):
    def __new__(cls, other):
        if isinstance(other, comfy.controlnet.ControlNet):
            other = copy.copy(other)
            other.__class__ = ControlNet
            return other
        raise ValueError(f"instance must be comfy.sd.ControlNet")

    def __init__(self, other):
        self.child_init = True

    def aitemplate_controlnet(
        self, latent_model_input, timesteps, encoder_hidden_states, controlnet_cond
    ):
        global AITemplateManger
        batch = latent_model_input.shape[0] / 2
        clip_chunks = int(encoder_hidden_states['c_crossattn'].shape[1] / 77)

        module_meta = ModuleMeta(os=AIT_OS, sd_version="v1", cuda_version=AIT_CUDA, batch_size=int(batch),
                                 width=latent_model_input.shape[3]*8, height=latent_model_input.shape[2]*8, model_type="controlnet", clip_chunks=int(clip_chunks), is_static=self.static_shape)
        # load the module
        module, is_cache = AITemplateManger.load_by_model_meta(module_meta)

        if not is_cache:
            AITemplateManger.loader.apply_controlnet(aitemplate_module=module,
                                                     controlnet=AITemplateManger.loader.compvis_controlnet(self.control_model.state_dict()))

        return controlnet_inference(
            exe_module=module,
            latent_model_input=latent_model_input,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
        )

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = ControlNet(self.previous_controlnet).get_control(x_noisy, t, cond, batched_number)
        # Moves inputs to GPU
        x_noisy = x_noisy.to(self.device)
        self.cond_hint_original = self.cond_hint_original.to(self.device)
        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(self.control_model.dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = comfy.controlnet.broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)
        # AITemplate inference, returns the same as regular
        control = self.aitemplate_controlnet(x_noisy, t, cond, self.cond_hint)
        control = list(control.values())
        out = {'middle': [], 'output': []}
        autocast_enabled = torch.is_autocast_enabled()

        for i in range(len(control)):
            if i == (len(control) - 1):
                key = 'middle'
                index = 0
            else:
                key = 'output'
                index = i
            x = control[i]
            if self.global_average_pooling:
                x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

            x *= self.strength
            if x.dtype != output_dtype and not autocast_enabled:
                x = x.to(output_dtype)

            if control_prev is not None and key in control_prev:
                prev = control_prev[key][index]
                if prev is not None:
                    x += prev
            out[key].append(x)
        if control_prev is not None and 'input' in control_prev:
            out['input'] = control_prev['input']
        return out

    def copy(self):
        c = ControlNet(self.control_model, global_average_pooling=self.global_average_pooling)
        c.set_is_static_shape(self.static_shape)
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        return c

    def set_is_static_shape(self, static_shape):
        self.static_shape = static_shape


class AitemplateBaseModel(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        self.dtype = torch.float16

        self.unet_ait_exe = None
        self.module_meta = ModuleMetaUnet(os=AIT_OS, cuda_version=AIT_CUDA, batch_size=(1, 1), have_control=False,
                                          width=(512, 512), height=(512, 512), clip_chunks=(1, 1), unnet_config=None)

    def set_base_model(self, base_model):
        self.base_model = base_model
        self.base_model_state_dict = base_model.state_dict()
        self.module_meta.unnet_config = base_model.model_config.unet_config

    def forward(self,  x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        batch_size = int(x.shape[0])
        clip_chunks = int(context.shape[1]/77)
        width = int(x.shape[3]*8)
        height = int(x.shape[2]*8)
        # print(batch_size, clip_chunks, width, height)
        if (self.unet_ait_exe == None
                    or self.module_meta.batch_size[0] > batch_size
                    or self.module_meta.batch_size[1] < batch_size
                    or self.module_meta.clip_chunks[0] > clip_chunks
                    or self.module_meta.clip_chunks[1] < clip_chunks
                    or self.module_meta.width[0] > width
                    or self.module_meta.width[1] < width
                    or self.module_meta.height[0] > height
                    or self.module_meta.height[1] < height
                    or self.module_meta.have_control != (control != None)
                ):
            if self.unet_ait_exe != None:
                del self.unet_ait_exe
                self.unet_ait_exe = None

            self.module_meta.batch_size = (batch_size, batch_size)
            self.module_meta.width = (width, width)
            self.module_meta.height = (height, height)
            self.module_meta.clip_chunks = (clip_chunks, clip_chunks)
            self.module_meta.have_control = (control != None)

            module_loader = AITLOADER.get_ait_module(self.module_meta)

            module_meta = module_loader.load_cache_exe()
            if module_meta == None:
                # origin_device = self.alphas_cumprod.device
                # self.to("cpu")
                module_meta = module_loader.build_exe(control=control)
                # self.to(origin_device)

            self.module_meta = module_meta

            print("apply_unet to unet_ait_exe")
            sd = self.base_model_state_dict
            keys = list(sd.keys())
            for k in keys:
                if not k.startswith("diffusion_model."):
                    sd.pop(k)
            sd = comfy.utils.state_dict_prefix_replace(sd, {"diffusion_model.": ""})
            module_loader.set_weights(sd)
            self.unet_ait_exe = module_loader

        return self.unet_ait_exe.apply_model(xc=x, t=timesteps, context=context, control=control, transformer_options=transformer_options, **kwargs)


class AITPatch:
    def __init__(self):
        self.ait_model = None

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        if hasattr(model_function.__self__, "hf_device_map"):
            return model_function(input_x, timestep_, **c)

        org_diffusion_model = model_function.__self__.diffusion_model

        if self.ait_model is None:
            self.ait_model = AitemplateBaseModel()

        self.ait_model.set_base_model(model_function.__self__)

        model_function.__self__.diffusion_model = self.ait_model
        try:
            result = model_function(input_x, timestep_, **c)
        finally:
            model_function.__self__.diffusion_model = org_diffusion_model
        return result

    def to(self, a):
        if self.ait_model is not None:
            if a == torch.device("cpu"):
                del self.ait_model
                self.ait_model = None
                print("unloaded AIT")
        return self


class ApplyAITemplateModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",), }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_aitemplate"

    CATEGORY = "loaders"

    def apply_aitemplate(self, model):
        patch = AITPatch()
        model_ait = model.clone()
        model_ait.set_model_unet_function_wrapper(patch)
        return (model_ait,)


class AitemplateAutoencoderKL(comfy.ldm.models.autoencoder.AutoencoderKL):
    @staticmethod
    def cast_from_AutoencoderKL(other, vae_data_type):
        if isinstance(other, comfy.ldm.models.autoencoder.AutoencoderKL):
            other.__class__ = AitemplateAutoencoderKL
            other.init_ait(vae_data_type)
            return other
        raise ValueError(f"instance must be comfy.ldm.models.autoencoder.AutoencoderKL")

    def cast_to_base_model(self):
        self.deinit_ait()
        self.__class__ = comfy.ldm.models.autoencoder.AutoencoderKL
        return self

    def init_ait(self, vae_data_type):
        self.ait_exe = None
        self.torch_compile_exe = None
        self.module_meta = ModuleMetaVAE(os=AIT_OS, cuda_version=AIT_CUDA, batch_size=(1, 1),
                                         width=(512, 512), height=(512, 512), vae_data_type=torch_dtype_to_string(vae_data_type))

    def deinit_ait(self):
        del self.ait_exe
        del self.module_meta
        del self.torch_compile_exe

    def decode(self, z):
        batch_size = int(z.shape[0])
        width = int(z.shape[3]*8)
        height = int(z.shape[2]*8)
        # print(batch_size, clip_chunks, width, height)
        if (self.ait_exe == None
                or self.module_meta.batch_size[0] > batch_size
                or self.module_meta.batch_size[1] < batch_size
                or self.module_meta.width[0] > width
                or self.module_meta.width[1] < width
                or self.module_meta.height[0] > height
                or self.module_meta.height[1] < height
                ):
            if self.ait_exe != None:
                del self.ait_exe

            self.module_meta.batch_size = (batch_size, batch_size)
            self.module_meta.width = (width, width)
            self.module_meta.height = (height, height)

            module_loader = AITLOADER.get_ait_module(self.module_meta)

            module_meta = module_loader.load_cache_exe()
            if module_meta == None:
                module_meta = module_loader.build_exe()

            self.module_meta = module_meta

            print("apply_vae to vae_ait_exe")
            module = module_loader.apply_ait_params(self.state_dict(), z.device)
            self.ait_exe = module
        return vae_inference(self.ait_exe, z, device=z.device, dtype=torch_dtype_to_string(z.dtype))


class AitemplateVAE(comfy.sd.VAE):
    @staticmethod
    def cast_from_VAE(other, keep_loaded):
        if isinstance(other, comfy.sd.VAE):
            other.__class__ = AitemplateVAE
            other.init_ait(keep_loaded)
            return other
        raise ValueError(f"instance must be comfy.sd.VAE")

    def init_ait(self, keep_loaded):
        self.keep_loaded = keep_loaded
        if self.keep_loaded:
            self.offload_device = self.device

    def decode(self, samples_in):
        if not (self.keep_loaded and type(self.first_stage_model) == AitemplateAutoencoderKL):
            self.first_stage_model = AitemplateAutoencoderKL.cast_from_AutoencoderKL(self.first_stage_model.to(self.device), self.vae_dtype)
        pixel_samples = super().decode(samples_in)
        if not self.keep_loaded:
            self.first_stage_model = self.first_stage_model.cast_to_base_model()
        return pixel_samples


class ApplyAITemplateVae:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE",),
                             "keep_loaded": ("BOOLEAN", {"default": True}), }}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "apply_aitemplate"

    CATEGORY = "loaders"

    def apply_aitemplate(self, vae, keep_loaded):
        vae = AitemplateVAE.cast_from_VAE(copy.copy(vae), keep_loaded)
        return (vae,)
