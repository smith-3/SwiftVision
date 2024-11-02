import tempfile
from fastapi import UploadFile, HTTPException, File, Depends
from sqlalchemy.orm import Session
from app.Segment import SAM
from utils.compress_descompress import compress_encoded_matrix, run_length_encode_matrix
from database.crud import CRUDOperations
from database.schemas import ProjectCreate, ImageCreate, MaskCreate
from fastapi.responses import JSONResponse
import numpy as np
import os

class ModelsAI:
    def __init__(self, db: Session):
        self.crud = CRUDOperations(db)
        self.sam = SAM()

    def process_image_and_generate_masks(self, user_id: int, project_name: str, file: UploadFile):
        try:
            # Leer los bytes de la imagen
            image_bytes = file.file.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Archivo de imagen vacío.")

            # Crear proyecto en la base de datos
            project_data = ProjectCreate(name=project_name)
            db_project = self.crud.create_project(project=project_data, user_id=user_id)

            # Crear imagen en la base de datos con los bytes de la imagen
            image_data = ImageCreate(base_image=image_bytes)
            db_image = self.crud.create_image(image=image_data, project_id=db_project.id)

            # Guardar la imagen en un archivo temporal para procesarla con SAM
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
                temp_image_file.write(image_bytes)
                temp_image_path = temp_image_file.name

            try:
                # Cargar imagen y generar máscaras
                segmentation_image = self.sam.load_image(temp_image_path)
                masks = self.sam.generate_masks(segmentation_image)
            finally:
                # Eliminar el archivo temporal
                os.remove(temp_image_path)

            if masks is None:
                raise HTTPException(status_code=500, detail="Error al generar las máscaras.")

            # Procesar máscaras y guardarlas en la base de datos
            masks_data = []

            for mask in masks:
                segmentation_data = mask.get("segmentation", None)
                if isinstance(segmentation_data, np.ndarray):
                    counts = np.array(segmentation_data, dtype=bool)
                    encoded_matrix = run_length_encode_matrix(counts)
                    mask_efficient = compress_encoded_matrix(encoded_matrix)
                    segmentation_data_dict = {
                        "counts": str(mask_efficient),  # Convertir a string
                        "size": str(segmentation_data.shape[::-1])  # Convertir a string para almacenarlo
                    }
                else:
                    segmentation_data_dict = {
                        "counts": mask.get("counts", []),
                        "size": mask.get("size", "N/A")
                    }
                bbox_str = str(mask.get("bbox", []))  # Convertir a string
                point_coords_str = str(mask.get("point_coords", []))
                # Guardar la máscara en la base de datos
                mask_data = MaskCreate(
                    counts=segmentation_data_dict["counts"],
                    size=segmentation_data_dict["size"],
                    bbox=bbox_str,
                    point_coords=point_coords_str
                )
                mask_data = self.crud.create_mask(annotation=mask_data, image_id=db_image.id)
                masks_data.append({
                    "id": mask_data.id,  # O el atributo que necesites
                    "counts": mask_data.counts,
                    "size": mask_data.size,
                    "bbox": mask_data.bbox,
                    "point_coords": mask_data.point_coords
                })
            print("Masks Data:", masks_data)  # Imprimir para depuración

            return JSONResponse(content={"masks": masks_data})

        except HTTPException as he:
            raise he  # Re-raise HTTPException to be handled by FastAPI
        except Exception as e:
            print(f"Error en process_image_and_generate_masks: {e}")  # Imprimir error en consola
            raise HTTPException(status_code=500, detail=str(e))
