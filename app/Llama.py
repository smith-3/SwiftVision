import os
import numpy as np
import torch
import yaml

import glob
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf

from modelsAI.lama.saicinpainting.evaluation.utils import move_to_device
from modelsAI.lama.saicinpainting.training.trainers import load_checkpoint
from modelsAI.lama.saicinpainting.evaluation.data import pad_tensor_to_modulo

from utils import load_img_to_array, save_array_to_img


class LamaInpaint:
    def __init__(
        self, lama_config: str, lama_ckpt: str, output_dir: str, device="cuda"
    ):
        self.lama_config = lama_config
        self.lama_ckpt = lama_ckpt
        self.output_dir = Path(output_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    def load_model(self):
        predict_config = OmegaConf.load(self.lama_config)
        predict_config.model.path = self.lama_ckpt

        train_config_path = os.path.join(predict_config.model.path, "config.yaml")
        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        checkpoint_path = os.path.join(
            predict_config.model.path, "models", predict_config.model.checkpoint
        )
        model = load_checkpoint(train_config, checkpoint_path, strict=False)
        model.to(self.device)
        model.freeze()
        return model

    @torch.no_grad()
    def inpaint_img(self, img: np.ndarray, mask: np.ndarray, model, mod=8):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255

        img = torch.from_numpy(img).float().div(255.0)
        mask = torch.from_numpy(mask).float()

        batch = {}
        batch["image"] = img.permute(2, 0, 1).unsqueeze(0)
        batch["mask"] = mask[None, None]
        unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
        batch["image"] = pad_tensor_to_modulo(batch["image"], mod)
        batch["mask"] = pad_tensor_to_modulo(batch["mask"], mod)

        batch = move_to_device(batch, self.device)
        batch["mask"] = (batch["mask"] > 0) * 1

        batch = model(batch)
        cur_res = batch["inpainted"][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        return cur_res

    def process_images(self, input_img_path: str, mask_glob: str):
        img_stem = Path(input_img_path).stem
        mask_paths = sorted(glob.glob(mask_glob))
        out_dir = self.output_dir / img_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        img = load_img_to_array(input_img_path)
        model = self.load_model()

        for mask_path in mask_paths:
            mask = load_img_to_array(mask_path)
            img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_path).name}"
            img_inpainted = self.inpaint_img(img, mask, model)
            save_array_to_img(img_inpainted, img_inpainted_p)


# Uso de la clase
# Puedes instanciar la clase y usar los métodos como sigue:
if __name__ == "__main__":
    lama = LamaInpaint(
        lama_config="./modelsAI/lama/configs/prediction/default.yaml",
        lama_ckpt="big-lama",
        output_dir="results",
        device="cuda",
    )

    lama.process_images(
        input_img_path="output/image.png", mask_glob="output/image/mask*.png"
    )
