from diffusers import AutoencoderKL
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import device, nn



class Vae(nn.Module): 
    def __init__(self,
                 model_path: str = "models/Flux_vae",
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        self.flux_vae = AutoencoderKL.from_pretrained(model_path).requires_grad_(False).to(device)

    @torch.no_grad()
    def encode(self,
               image: str,
               transforms: transforms.Compose) -> str:
        image = Image.open(image).convert("RGB")
        image_tensor = transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoded = self.flux_vae.encode(image_tensor)
            latent = encoded.latent_dist.sample()
        return latent



