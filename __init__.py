from .AITemplate_node import ApplyAITemplateVae, ApplyAITemplateModel

NODE_CLASS_MAPPINGS = {
    "ApplyAITemplateVae": ApplyAITemplateVae,
    "ApplyAITemplateModel": ApplyAITemplateModel,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyAITemplateModel": "Apply AITemplate : Model (UNet)",
    "ApplyAITemplateVae": "Apply AITemplate : Vae",
}
