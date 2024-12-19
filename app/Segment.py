import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from modelsAI.segment_anything.segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor,
)
import logging

logger = logging.getLogger(__name__)

class SAM:
    def __init__(self, checkpoint_path: str = "./models/weights/mobile_sam.pt"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = None
        self.mask_generator = None
        self.predictor = None
        self.masks_data = None
        self.setup_model()
        self.print_versions()

    def print_versions(self):
        """Print versions of PyTorch, Torchvision, and CUDA availability."""
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Torchvision version: {torchvision.__version__}")
        logger.info(f"CUDA is available: {torch.cuda.is_available()}")

    def load_image(self, image_path: str):
        """Load an image from the given path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        return image

    def setup_model(self, model_type="vit_t"):  # vit_h para sam normal
        """Set up the SAM model and the mask generator."""
        self.sam = sam_model_registry[model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=8,  # 8-32
            pred_iou_thresh=0.9,  # 0.86
            stability_score_thresh=0.95,  # 0.92
            crop_n_layers=0,  # 1
            crop_n_points_downscale_factor=2,
            min_mask_region_area=500,  # Requires open-cv to run post-processing 100
        )
        self.predictor = SamPredictor(self.sam)
        logger.info("SAM model initialized successfully.")

    def generate_masks(self, image: np.ndarray):
        """Generate masks for the given image."""
        if image is None:
            raise ValueError("No image provided. Please provide an image.")
        masks = self.mask_generator.generate(image)
        sorted_anns = sorted(masks, key=lambda x: x["area"])
        self.masks_data = sorted_anns
        logger.info(f"Number of masks generated: {len(masks)}")
        return self.masks_data

    def predict_mask_with_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = True,
    ):
        """Predict masks based on points and labels."""
        try:
            # Validaciones
            if not isinstance(points, np.ndarray):
                raise TypeError(f"Expected points to be a NumPy array, but got {type(points)}")
            if not isinstance(labels, np.ndarray):
                raise TypeError(f"Expected labels to be a NumPy array, but got {type(labels)}")
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(f"Points array should have shape (N, 2), but got {points.shape}")
            if labels.ndim != 1 or len(labels) != len(points):
                raise ValueError(f"Labels array should be of length {len(points)}, but got {len(labels)}")

            # Realizar la predicción
            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=multimask_output,
            )
            return masks
        except (TypeError, ValueError) as e:
            logger.error(f"Error in predict_mask_with_points: {e}")
            raise
        except Exception as e:
            logger.exception("Unexpected error in predict_mask_with_points.")
            raise
