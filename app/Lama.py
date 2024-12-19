import os
import torch
import yaml
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from modelsAI.lama.saicinpainting.training.trainers import load_checkpoint
from modelsAI.lama.saicinpainting.evaluation.data import pad_tensor_to_modulo
from modelsAI.lama.saicinpainting.evaluation.utils import move_to_device
from utils import load_img_to_array, save_array_to_img
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class Lama:
    def __init__(self, device="cuda"):
        self.config_path = "./modelsAI/lama/configs/prediction/default.yaml"
        self.checkpoint_path = "./models/pretrained_models/big-lama"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        logger.info("Lama model loaded successfully.")

    def _build_model(self):
        try:
            predict_config = OmegaConf.load(self.config_path)
            predict_config.model.path = self.checkpoint_path

            train_config_path = os.path.join(predict_config.model.path, "config.yaml")
            with open(train_config_path, "r") as f:
                train_config = OmegaConf.create(yaml.safe_load(f))

            train_config.training_model.predict_only = True
            train_config.visualizer.kind = "noop"

            checkpoint_path = os.path.join(
                predict_config.model.path, "models", predict_config.model.checkpoint
            )
            map_location = self.device

            model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=map_location)
            model.to(self.device)
            model.freeze()
            return model
        except Exception as e:
            logger.exception("Failed to build the Lama model.")
            raise

    @torch.no_grad()
    def inpaint(self, img: np.ndarray, mask: np.ndarray, mod=8):
        try:
            assert len(mask.shape) == 2, "La máscara debe ser 2D."
            if np.max(mask) == 1:
                mask = mask * 255

            img_tensor = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(self.device)

            unpad_to_size = [img_tensor.shape[2], img_tensor.shape[3]]
            img_padded = pad_tensor_to_modulo(img_tensor, mod)
            mask_padded = pad_tensor_to_modulo(mask_tensor, mod)
            mask_padded = (mask_padded > 0).float()

            batch = {
                "image": img_padded,
                "mask": mask_padded
            }

            batch = move_to_device(batch, self.device)

            output = self.model(batch)
            result = output["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()

            # Unpad the result to the original size
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                result = result[:orig_height, :orig_width]

            result = np.clip(result * 255, 0, 255).astype("uint8")
            return result
        except Exception as e:
            logger.exception("Error during inpainting.")
            raise
