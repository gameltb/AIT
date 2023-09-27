import os
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.sd

import torch
import copy

from .module.loader import AITLoader
from .module.unnet import ModuleMetaUnet
from .module.inference import unet_inference, controlnet_inference

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


class AITemplateVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "vae": ("VAE",),
                    "keep_loaded": (["enable", "disable"], ),
                    "samples": ("LATENT", ), "vae": ("VAE", )
                }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, vae, keep_loaded, samples):
        global AITemplateManger
        batch_number = 1

        module_meta = ModuleMeta(os=AIT_OS, sd_version="v1", cuda_version=AIT_CUDA, batch_size=batch_number,
                                 width=samples["samples"].shape[3]*8, height=samples["samples"].shape[2]*8, model_type="vae_decode")
        # load the module
        module, is_cache = AITemplateManger.load_by_model_meta(module_meta)

        if not is_cache:
            AITemplateManger.loader.apply_vae(aitemplate_module=module,
                                              vae=AITemplateManger.loader.compvis_vae(vae.first_stage_model.state_dict()),)

        samples_in = samples["samples"]

        pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
        for x in range(0, samples_in.shape[0], batch_number):
            samples = samples_in[x:x+batch_number]
            pixel_samples[x:x+batch_number] = torch.clamp((vae_inference(module, samples) + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()

        pixel_samples = pixel_samples.cpu().movedim(1, -1)

        if keep_loaded == "disable":
            AITemplateManger.unload(["vae_decode"])

        return (pixel_samples,)


class AitemplateBaseModel(comfy.model_base.BaseModel):
    @staticmethod
    def cast_from_base_model(other):
        if isinstance(other, comfy.model_base.BaseModel):
            other.__class__ = AitemplateBaseModel
            other.init_ait()
            return other
        raise ValueError(f"instance must be comfy.model_base.BaseModel")

    def cast_to_base_model(self):
        self.deinit_ait()
        self.__class__ = comfy.model_base.BaseModel
        return self

    def apply_model_compiled(self, x, t, c_concat=None, c_crossattn=None, c_adm=None, control=None, transformer_options={}):
        if c_concat is not None:
            xc = torch.cat([x] + [c_concat], dim=1)
        else:
            xc = x
        context = c_crossattn
        dtype = self.get_dtype()
        xc = xc.to(dtype)
        t = t.to(dtype)
        context = context.to(dtype)
        if c_adm is not None:
            c_adm = c_adm.to(dtype)

        if self.unet_torch_compile_exe == None or self.module_meta.batch_size != int(x.shape[0]) or self.module_meta.clip_chunks != int(c_crossattn.shape[1]/77):
            self.module_meta.batch_size = int(x.shape[0])
            self.module_meta.width = int(x.shape[3]*8)
            self.module_meta.height = int(x.shape[2]*8)
            self.module_meta.clip_chunks = int(c_crossattn.shape[1]/77)
            self.unet_torch_compile_exe = torch.compile(self.diffusion_model, fullgraph=True, backend="inductor", mode="default")
        return self.unet_torch_compile_exe(xc, t, context=context, y=c_adm, control=control, transformer_options=transformer_options).float()

    def init_ait(self):
        self.unet_ait_exe = None
        self.unet_torch_compile_exe = None
        self.module_meta = ModuleMetaUnet(os=AIT_OS, cuda_version=AIT_CUDA, batch_size=(1, 1), have_control=False,
                                          width=(512, 512), height=(512, 512), clip_chunks=(1, 1), unnet_config=self.model_config.unet_config)

    def deinit_ait(self):
        del self.unet_ait_exe
        del self.module_meta
        del self.unet_torch_compile_exe

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, c_adm=None, control=None, transformer_options={}):
        # if len(transformer_options) > 0:
        #     return self.apply_model_compiled(x, t, c_concat=c_concat, c_crossattn=c_crossattn, c_adm=c_adm, control=control, transformer_options=transformer_options)
        timesteps_pt = t
        latent_model_input = x
        encoder_hidden_states = None
        down_block_residuals = None
        mid_block_residual = None
        add_embeds = None
        if c_crossattn is not None:
            encoder_hidden_states = c_crossattn
        if c_concat is not None:
            latent_model_input = torch.cat([x] + [c_concat], dim=1)
        if control is not None:
            down_block_residuals = control["output"]
            mid_block_residual = control["middle"][0]
        if c_adm is not None:
            add_embeds = c_adm
        batch_size = int(latent_model_input.shape[0])
        clip_chunks = int(encoder_hidden_states.shape[1]/77)
        width = int(latent_model_input.shape[3]*8)
        height = int(latent_model_input.shape[2]*8)
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
            self.module_meta.batch_size = (batch_size, batch_size)
            self.module_meta.width = (width, width)
            self.module_meta.height = (height, height)
            self.module_meta.clip_chunks = (clip_chunks, clip_chunks)
            self.module_meta.have_control = (control != None)

            module_loader = AITLOADER.get_ait_module(self.module_meta)

            module_meta = module_loader.load_cache_exe()
            if module_meta == None:
                origin_device = self.alphas_cumprod.device
                self.to("cpu")
                module_meta = module_loader.build_exe()
                self.to(origin_device)

            self.module_meta = module_meta

            print("apply_unet to unet_ait_exe")
            module = module_loader.apply_ait_params(self.state_dict(), self.alphas_cumprod.device)
            self.unet_ait_exe = module

        return unet_inference(
            self.unet_ait_exe,
            latent_model_input=latent_model_input,
            timesteps=timesteps_pt,
            encoder_hidden_states=encoder_hidden_states,
            down_block_residuals=down_block_residuals,
            mid_block_residual=mid_block_residual,
            add_embeds=add_embeds,
        )


class AitemplateModelPatcher(comfy.model_patcher.ModelPatcher):
    @staticmethod
    def cast_from_model_patcher(other):
        if isinstance(other, comfy.model_patcher.ModelPatcher):
            other.__class__ = AitemplateModelPatcher
            return other
        raise ValueError(f"instance must be comfy.model_patcher.ModelPatcher")

    def clone(self):
        n = AitemplateModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]

        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        return n

    def patch_model(self, device_to=None):
        super().patch_model(device_to)
        print("patch_model ", device_to)
        if type(self.model) == comfy.model_base.BaseModel:
            self.model = AitemplateBaseModel.cast_from_base_model(self.model)

    def unpatch_model(self, device_to=None):
        if type(self.model) == AitemplateBaseModel:
            self.model = self.model.cast_to_base_model()
        super().unpatch_model(device_to)
        print("unpatch_model ", device_to)


class ApplyAITemplate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             #  "keep_loaded": (["enable", "disable"], ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_aitemplate"

    CATEGORY = "loaders"

    def apply_aitemplate(self, model):
        model = AitemplateModelPatcher.cast_from_model_patcher(model.clone())
        return (model,)
