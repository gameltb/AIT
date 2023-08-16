from .AITemplate_node import AITemplateVAEDecode, ApplyAITemplate

NODE_CLASS_MAPPINGS = {
    "AITemplateVAEDecode": AITemplateVAEDecode,
    "ApplyAITemplate": ApplyAITemplate,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AITemplateVAEDecode": "VAE Decode (AITemplate)",
    "ApplyAITemplate": "ApplyAITemplate",
}
