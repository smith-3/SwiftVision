import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import logging
from utils.mask_processing import crop_for_filling_pre, crop_for_filling_post
from utils.crop_for_replacing import recover_size, resize_and_pad
from utils import load_img_to_array, save_array_to_img
import numpy as np

logger = logging.getLogger(__name__)

class StableDiffusion:
    def __init__(self, checkpoint: str = "stabilityai/stable-diffusion-2-inpainting", device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pipe = self._load_pipeline()
        logger.info("Stable Diffusion pipeline loaded successfully.")

    def _load_pipeline(self):
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.checkpoint,
                torch_dtype=torch.float32,
            ).to(self.device)
            return pipe
        except Exception as e:
            logger.exception("Failed to load Stable Diffusion pipeline.")
            raise

    def fill_img_with_sd(self, img: np.ndarray, mask: np.ndarray, text_prompt: str, step: int = 50):
        try:
            img_crop, mask_crop = crop_for_filling_pre(img, mask)
            img_crop_filled = self.pipe(
                prompt=text_prompt,
                image=Image.fromarray(img_crop),
                mask_image=Image.fromarray(mask_crop),
                num_inference_steps=step
            ).images[0]
            img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
            return img_filled
        except Exception as e:
            logger.exception("Error in fill_img_with_sd.")
            raise

    def replace_img_with_sd(self, img: np.ndarray, mask: np.ndarray, text_prompt: str, step: int = 50):
        try:
            img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
            img_padded_filled = self.pipe(
                prompt=text_prompt,
                image=Image.fromarray(img_padded),
                mask_image=Image.fromarray(255 - mask_padded),
                num_inference_steps=step,
            ).images[0]
            height, width, _ = img.shape
            img_resized, mask_resized = recover_size(
                np.array(img_padded_filled), mask_padded, (height, width), padding_factors)
            mask_resized = np.expand_dims(mask_resized, -1) / 255
            img_resized = img_resized * (1 - mask_resized) + img * mask_resized
            return img_resized
        except Exception as e:
            logger.exception("Error in replace_img_with_sd.")
            raise
