import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from models_ai.segment_anything.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
def es_color_oscuro(r, g, b):
    luminancia = 0.299 * r + 0.587 * g + 0.114 * b
    return luminancia < 128

class ImageSegmentation:
    def __init__(self, checkpoint_path: str = "./models/weights/mobile_sam.pt"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image = None
        self.sam = None
        self.mask_generator = None
        self.predictor = None
        self.masks_data = None
        self.print_versions()
        self.setup_model()

    def print_versions(self):
        """Print versions of PyTorch, Torchvision, and CUDA availability."""
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())

    def load_image(self, image_path: str):
        """Load an image from the given path."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        plt.axis('off')
        #plt.show()

    def setup_model(self, model_type = "vit_t"): #vit_h para sam normal
        """Set up the SAM model and the mask generator."""
        self.sam = sam_model_registry[model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=8,#8-32
            pred_iou_thresh=0.9, #0.86
            stability_score_thresh=0.95, #0.92
            crop_n_layers=1,#1
            crop_n_points_downscale_factor=2,
            min_mask_region_area=500,  # Requires open-cv to run post-processing 100
        )
        self.predictor = SamPredictor(self.sam)

    def generate_masks(self):
        """Generate masks for the loaded image."""
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")
        masks = self.mask_generator.generate(self.image)
        sorted_anns = sorted(masks, key=(lambda x: x['area']))
        self.masks_data = sorted_anns
        print(f"Number of masks generated: {len(masks)}")
        return self.masks_data

    def predict_mask_with_box(self, box: np.ndarray, multimask_output: bool = False):
        """Predict masks based on a bounding box."""
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=multimask_output,
        )
        return masks, scores, logits

    def predict_mask_with_points(self, points: np.ndarray, labels: np.ndarray, multimask_output: bool = True):
        """Predict masks based on points and labels."""
        try:
            # Verificar que los puntos y las etiquetas son arrays de NumPy
            if not isinstance(points, np.ndarray):
                raise TypeError(f"Expected points to be a NumPy array, but got {type(points)}")
            if not isinstance(labels, np.ndarray):
                raise TypeError(f"Expected labels to be a NumPy array, but got {type(labels)}")

            # Verificar dimensiones adecuadas de los arrays
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(f"Points array should have shape (N, 2), but got {points.shape}")
            if labels.ndim != 1 or len(labels) != len(points):
                raise ValueError(f"Labels array should be of length {len(points)}, but got {len(labels)}")

            # Realizar la predicción
            self.predictor.set_image(self.image)
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=multimask_output,
            )
            return masks

        except TypeError as te:
            print(f"Type Error in predict_mask_with_points: {te}")
            raise

        except ValueError as ve:
            print(f"Value Error in predict_mask_with_points: {ve}")
            raise

        except Exception as e:
            print(f"Unexpected error in predict_mask_with_points: {e}")
            raise


    def refine_mask_with_additional_prompts(self, points: np.ndarray, labels: np.ndarray, mask_input: np.ndarray, multimask_output: bool = False):
        """Refine the mask using additional prompts."""
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=mask_input[None, :, :],
            multimask_output=multimask_output,
        )
        return masks, scores, logits


if __name__ == "__main__":
    # Usage of the ImageSegmentation class
    image_path = './images/peloCorto.jpeg'
    segmentation = ImageSegmentation()
    segmentation.load_image(image_path)
    masks = segmentation.generate_masks()
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

     # Assuming you want to change the color of the first mask to red and save it
    hair_mask = sorted_anns[2]
    segmentation_mask = hair_mask['segmentation']

    # Refine the mask
    refined_mask = segmentation.refine_hair_mask(segmentation_mask)



    red_color = (213,196,161)
    output_image_path = "./response/output.png"

    segmentation.hair_color(refined_mask, red_color, output_image_path)
    #segmentation.show_anns(sorted_anns)

