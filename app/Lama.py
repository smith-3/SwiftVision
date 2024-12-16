import os
import sys
import numpy as np
import torch
import yaml
import glob
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path

from modelsAI.lama.saicinpainting.evaluation.utils import move_to_device
from modelsAI.lama.saicinpainting.training.trainers import load_checkpoint
from modelsAI.lama.saicinpainting.evaluation.data import pad_tensor_to_modulo
from utils import load_img_to_array, save_array_to_img

class Lama:
    def __init__(self, device="cuda"):
        self.config_path = "./modelsAI/lama/configs/prediction/default.yaml"
        self.checkpoint_path = "./models/pretrained_models/big-lama"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

    def _build_model(self):
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
        map_location = torch.device("cpu") if self.device.type == "cpu" else None
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=map_location)
        model.to(self.device)
        model.freeze()
        return model
    

    @torch.no_grad()
    def inpaint(self, img: np.ndarray, mask: np.ndarray, mod=8):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(img).float().div(255.0)
        mask = torch.from_numpy(mask).float()
        predict_config = OmegaConf.load(self.config_path)
        predict_config.model.path = self.checkpoint_path
        # device = torch.device(predict_config.device)
        # device = torch.device(device)
        device = torch.device("cpu")

        train_config_path = os.path.join(predict_config.model.path, "config.yaml")

        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        checkpoint_path = os.path.join(
            predict_config.model.path, "models", predict_config.model.checkpoint
        )
        model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location="cpu"
        )
        model.freeze()
        if not predict_config.get("refine", False):
            model.to(device)

        batch = {}
        batch["image"] = img.permute(2, 0, 1).unsqueeze(0)
        batch["mask"] = mask[None, None]
        unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
        batch["image"] = pad_tensor_to_modulo(batch["image"], mod)
        batch["mask"] = pad_tensor_to_modulo(batch["mask"], mod)
        batch = move_to_device(batch, device)
        batch["mask"] = (batch["mask"] > 0) * 1

        batch = model(batch)
        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        return cur_res
    
    # @torch.no_grad()
    # def inpaint(self, img: np.ndarray, mask: np.ndarray, mod=8):
    #     print("matriz imagen")
    #     print(img)
    #     print("matriz mask")
    #     print(mask)

    #     assert len(mask.shape) == 2
    #     if np.max(mask) == 1:
    #         mask = mask * 255

    #     img = torch.from_numpy(img).float().div(255.0)
    #     mask = torch.from_numpy(mask).float()

    #     batch = {
    #         "image": img.permute(2, 0, 1).unsqueeze(0),
    #         "mask": mask[None, None]
    #     }
        
    #     unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
    #     batch["image"] = pad_tensor_to_modulo(batch["image"], mod)
    #     batch["mask"] = pad_tensor_to_modulo(batch["mask"], mod)
    #     batch = move_to_device(batch, self.device)
    #     batch["mask"] = (batch["mask"] > 0) * 1

    #     batch = self.model(batch)
    #     cur_res = batch["inpainted"][0].permute(1, 2, 0)
    #     cur_res = cur_res.detach().cpu().numpy()

    #     if unpad_to_size is not None:
    #         orig_height, orig_width = unpad_to_size
    #         cur_res = cur_res[:orig_height, :orig_width]

    #     cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    #     return cur_res

    def process(self, input_img: str, input_mask_glob: str, output_dir: str):
        img_stem = Path(input_img).stem
        mask_paths = sorted(glob.glob(input_mask_glob))
        output_path = Path(output_dir) / img_stem
        output_path.mkdir(parents=True, exist_ok=True)

        img = load_img_to_array(input_img)
        for mask_path in mask_paths:
            mask = load_img_to_array(mask_path)
            result = self.inpaint(img, mask)

            output_file = output_path / f"inpainted_with_{Path(mask_path).name}"
            save_array_to_img(result, output_file)

if __name__ == "__main__":


    input_img = "FA_demo/FA1_dog.png"
    input_mask_glob = "results/FA1_dog/mask*.png"
    output_dir = "results"
    lama = Lama()
    lama.process(input_img, input_mask_glob, output_dir)
