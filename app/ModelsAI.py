# ModelsAI.py

import tempfile
import os
import io
import json
import logging
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from fastapi import UploadFile, HTTPException, File, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.Lama import Lama
from app.Segment import SAM
from app.StableDiffusion import StableDiffusion
from utils.compress_descompress import (
    compress_encoded_matrix,
    decompress_encoded_matrix,
    run_length_encode_matrix
)
from utils.utils import dilate_mask
from database.crud import CRUDOperations
from database.schemas import ProjectCreate, ImageCreate, MaskCreate

# Configurar el logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelsAI:
    def __init__(self, db: Session):
        self.crud = CRUDOperations(db)
        self.sam = SAM()
        self.lama = Lama()
        self.stable_diffusion = StableDiffusion()

    def _get_project_image_mask(
        self, project_id: int, image_id: int, mask_id: int
    ) -> Dict[str, Any]:
        """
        Recupera el proyecto, imagen y máscara de la base de datos.
        """
        db_project = self.crud.get_project(project_id)
        db_image = self.crud.get_image(image_id)
        db_mask = self.crud.get_mask(mask_id)

        if not all([db_project, db_image, db_mask]):
            logger.error(
                f"Proyecto, imagen o máscara no encontrados. "
                f"Proyecto ID: {project_id}, Imagen ID: {image_id}, Máscara ID: {mask_id}"
            )
            raise HTTPException(
                status_code=404,
                detail="Proyecto, imagen o máscara no encontrados."
            )

        return {
            "project": db_project,
            "image": db_image,
            "mask": db_mask
        }

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Convierte bytes de imagen a un array NumPy en formato RGB.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_array = np.array(img, dtype=np.uint8)

            if img_array.ndim != 3 or img_array.shape[2] != 3:
                logger.error("La imagen no tiene el formato esperado (RGB).")
                raise HTTPException(
                    status_code=400,
                    detail="La imagen no tiene el formato esperado (RGB)."
                )

            return img_array
        except Exception as e:
            logger.exception("Error al cargar la imagen.")
            raise HTTPException(
                status_code=400,
                detail="Error al procesar la imagen."
            )

    def load_mask(self, mask_counts: List[List[List[int]]], mask_size: List[int]) -> np.ndarray:
        """
        Decodifica y dilata la máscara.
        """
        try:
            # Verificar que mask_size tenga exactamente dos elementos
            if len(mask_size) != 2:
                raise ValueError("mask_size debe contener exactamente dos elementos: [width, height]")

            width, height = mask_size  # [width, height]
            logger.info(f"Mask size: width={width}, height={height}")

            # Verificar que mask_counts esté bien formateado
            for i, row in enumerate(mask_counts):
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError(f"mask_counts contiene filas mal formateadas en el índice {i}.")

            # Cambiar el orden de las dimensiones al pasar a decompress_encoded_matrix
            mask = decompress_encoded_matrix(mask_counts, (height, width))  # (height, width)
            logger.info(f"Descompresión de la máscara exitosa con shape {mask.shape}")

            mask = dilate_mask(mask, 30)

            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            if mask.ndim != 2:
                logger.error("La máscara no tiene el formato esperado (2D).")
                raise HTTPException(
                    status_code=400,
                    detail="La máscara no tiene el formato esperado (2D)."
                )
            if np.max(mask) == 1:
                mask *= 255

            return mask
        except ValueError as e:
            logger.error(f"Error en load_mask: {e}")
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        except Exception as e:
            logger.exception("Error al procesar la máscara.")
            raise HTTPException(
                status_code=500,
                detail="Error interno al procesar la máscara."
            )

    def _validate_dimensions(self, img: np.ndarray, mask: np.ndarray):
        """
        Valida que las dimensiones de la imagen y la máscara coincidan.
        """
        if img.shape[:2] != mask.shape:
            logger.error("La imagen y la máscara tienen dimensiones diferentes.")
            raise HTTPException(
                status_code=400,
                detail="La imagen y la máscara tienen dimensiones diferentes."
            )

    def _save_image_locally(self, img: np.ndarray, filename: str, directory: str):
        """
        Guarda una imagen localmente en el directorio especificado.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            img_pil = Image.fromarray(img)
            img_path = os.path.join(directory, filename)
            img_pil.save(img_path)
            logger.info(f"Imagen guardada en {img_path}")
        except Exception as e:
            logger.exception("Error al guardar la imagen localmente.")
            raise HTTPException(
                status_code=500,
                detail="Error al guardar la imagen localmente."
            )

    def _process_inpainted_image(self, img_inpainted: np.ndarray, original_image_id: int, db_project: Any) -> Dict[str, int]:
        """
        Procesa la imagen inpainted: la convierte a bytes, la guarda en la base de datos y localmente.
        Además, genera las máscaras asociadas a esta nueva imagen.
        """
        try:
            img_inpainted = img_inpainted.astype(np.uint8)
            img_pil = Image.fromarray(img_inpainted)

            # Convertir a bytes
            byte_io = io.BytesIO()
            img_pil.save(byte_io, format='PNG')
            img_bytes = byte_io.getvalue()

            # Guardar en la base de datos
            inpainted_image_data = ImageCreate(base_image=img_bytes)
            inpainted_image = self.crud.create_image(image=inpainted_image_data, project_id=db_project.id)

            # Guardar localmente
            output_dir = "./inpainted_images"
            filename = f"inpainted_image_{original_image_id}.png"
            self._save_image_locally(img_inpainted, filename, output_dir)

            logger.info(f"Imagen inpainted creada con ID: {inpainted_image.id}")

            # Generar máscaras para la nueva imagen
            masks_generated = self.generate_masks_for_image(inpainted_image.id)
            logger.info(f"Máscaras generadas para la imagen inpainted ID {inpainted_image.id}: {masks_generated}")

            return {"inpainted_image_id": inpainted_image.id}

        except Exception as e:
            logger.exception("Error al procesar la imagen inpainted.")
            raise HTTPException(
                status_code=500,
                detail="Error al procesar la imagen inpainted."
            )

    def _common_inpaint_process(
        self, 
        project_id: int, 
        image_id: int, 
        mask_id: int, 
        prompt: str, 
        inpaint_method: str
    ) -> Dict[str, int]:
        """
        Método común para procesar inpainting utilizando diferentes métodos.
        """
        data = self._get_project_image_mask(project_id, image_id, mask_id)
        img = self._load_image(data["image"].base_image)

        # Llamar correctamente al método `load_mask`
        mask = self.load_mask(data["mask"].counts, data["mask"].size)

        self._validate_dimensions(img, mask)

        self.save_images(img, mask)

        if inpaint_method == "stable_diffusion":
            img_inpainted = self.stable_diffusion.fill_img_with_sd(img, mask, prompt)
        elif inpaint_method == "replace_sd":
            img_inpainted = self.stable_diffusion.replace_img_with_sd(img, mask, prompt)
        elif inpaint_method == "lama":
            img_inpainted = self.lama.inpaint(img, mask)
        else:
            logger.error(f"Método de inpainting desconocido: {inpaint_method}")
            raise HTTPException(
                status_code=400,
                detail="Método de inpainting desconocido."
            )

        return self._process_inpainted_image(img_inpainted, image_id, data["project"])

    def generate_image(self, project_id: int, image_id: int, mask_id: int, prompt: str) -> Dict[str, int]:
        """
        Genera una imagen inpainted utilizando Stable Diffusion.
        """
        return self._common_inpaint_process(project_id, image_id, mask_id, prompt, "stable_diffusion")

    def generate_background(self, project_id: int, image_id: int, mask_id: int, prompt: str) -> Dict[str, int]:
        """
        Genera un fondo inpainted utilizando Stable Diffusion.
        """
        return self._common_inpaint_process(project_id, image_id, mask_id, prompt, "replace_sd")

    def process_inpainting_with_mask(self, project_id: int, image_id: int, mask_id: int) -> Dict[str, int]:
        """
        Procesa inpainting utilizando el método Lama.
        """
        return self._common_inpaint_process(project_id, image_id, mask_id, prompt="", inpaint_method="lama")

    def save_images(self, img: np.ndarray, mask: np.ndarray):
        """
        Guarda la imagen y la máscara localmente.
        """
        try:
            os.makedirs("./saved_images", exist_ok=True)

            # Generar nombres únicos usando UUID para evitar sobrescrituras
            import uuid
            img_filename = f"image_{uuid.uuid4()}.png"
            mask_filename = f"mask_{uuid.uuid4()}.png"

            # Guardar la imagen
            self._save_image_locally(img, img_filename, "./saved_images")

            # Guardar la máscara
            self._save_image_locally(mask, mask_filename, "./saved_images")
        except Exception as e:
            logger.exception("Error al guardar las imágenes.")
            raise HTTPException(
                status_code=500,
                detail="Error al guardar las imágenes."
            )

    def process_image_and_generate_masks(
        self, 
        user_id: int, 
        project_name: str, 
        file: UploadFile
    ) -> JSONResponse:
        """
        Sube una imagen, genera máscaras y las guarda en la base de datos.
        """
        try:
            # Leer los bytes de la imagen
            image_bytes = file.file.read()
            if not image_bytes:
                logger.error("Archivo de imagen vacío.")
                raise HTTPException(
                    status_code=400,
                    detail="Archivo de imagen vacío."
                )

            # Crear proyecto en la base de datos
            project_data = ProjectCreate(name=project_name)
            db_project = self.crud.create_project(project=project_data, user_id=user_id)
            logger.info(f"Proyecto creado con ID: {db_project.id}")

            # Crear imagen en la base de datos con los bytes de la imagen
            image_data = ImageCreate(base_image=image_bytes)
            db_image = self.crud.create_image(image=image_data, project_id=db_project.id)
            logger.info(f"Imagen creada con ID: {db_image.id}")

            # Guardar la imagen en un archivo temporal para procesarla con SAM
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
                temp_image_file.write(image_bytes)
                temp_image_path = temp_image_file.name
                logger.info(f"Imagen temporal guardada en: {temp_image_path}")

            try:
                # Cargar imagen y generar máscaras
                segmentation_image = self.sam.load_image(temp_image_path)
                masks = self.sam.generate_masks(segmentation_image)
                logger.info(f"Mascaras generadas: {len(masks) if masks else 0}")
            finally:
                # Eliminar el archivo temporal
                os.remove(temp_image_path)
                logger.info(f"Archivo temporal eliminado: {temp_image_path}")

            if not masks:
                logger.error("Error al generar las máscaras.")
                raise HTTPException(
                    status_code=500,
                    detail="Error al generar las máscaras."
                )

            # Procesar y guardar máscaras en la base de datos
            masks_data = self._process_and_save_masks(masks, db_image.id)
            logger.info(f"Mascaras guardadas: {masks_data}")

            return JSONResponse(content={"id": db_project.id})

        except HTTPException as he:
            raise he  # Re-raise HTTPException to be handled by FastAPI
        except Exception as e:
            logger.exception("Error en process_image_and_generate_masks.")
            raise HTTPException(
                status_code=500,
                detail="Error interno al procesar la imagen y generar máscaras."
            )

    def _process_and_save_masks(self, masks: List[Dict[str, Any]], image_id: int) -> List[Dict[str, Any]]:
        """
        Procesa las máscaras generadas y las guarda en la base de datos.
        """
        masks_data = []
        try:
            for mask in masks:
                segmentation_data = mask.get("segmentation")
                if isinstance(segmentation_data, np.ndarray):
                    counts = run_length_encode_matrix(segmentation_data.astype(bool))
                    compressed_counts = compress_encoded_matrix(counts)

                    # Convertir las tuplas en listas serializables
                    serializable_counts = [
                        [[list(length_value) for length_value in row], count]
                        for row, count in compressed_counts
                    ]

                    mask_size = list(segmentation_data.shape[::-1])  # Convertir a lista [width, height]
                    logger.info(f"Mask size (width, height): {mask_size}")
                else:
                    raise ValueError("El formato de la máscara no es válido.")

                # Crear la máscara en la base de datos
                mask_data = MaskCreate(
                    counts=serializable_counts,  # Estructura serializable
                    size=mask_size,              # Lista [width, height]
                    bbox=mask.get("bbox", []),   # Lista serializable
                    point_coords=mask.get("point_coords", [])  # Lista serializable
                )

                mask_record = self.crud.create_mask(mask=mask_data, image_id=image_id)
                masks_data.append({
                    "id": mask_record.id,
                    "counts": mask_record.counts,
                    "size": mask_record.size,
                    "bbox": mask_record.bbox,
                    "point_coords": mask_record.point_coords
                })
                logger.debug(f"Mask record saved: {mask_record.id}")
            return masks_data
        except Exception as e:
            logger.exception("Error al procesar y guardar las máscaras.")
            raise HTTPException(
                status_code=500,
                detail="Error al procesar y guardar las máscaras."
            )

    def generate_masks_for_image(self, image_id: int) -> List[Dict[str, Any]]:
        """
        Genera máscaras para una imagen existente utilizando SAM y las guarda en la base de datos.
        """
        try:
            # Recuperar la imagen de la base de datos
            db_image = self.crud.get_image(image_id)
            if not db_image:
                logger.error(f"Imagen no encontrada con ID: {image_id}")
                raise HTTPException(
                    status_code=404,
                    detail="Imagen no encontrada."
                )

            # Guardar la imagen en un archivo temporal para procesarla con SAM
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
                temp_image_file.write(db_image.base_image)
                temp_image_path = temp_image_file.name
                logger.info(f"Imagen temporal para generación de máscaras guardada en: {temp_image_path}")

            try:
                # Cargar imagen y generar máscaras
                segmentation_image = self.sam.load_image(temp_image_path)
                masks = self.sam.generate_masks(segmentation_image)
                logger.info(f"Mascaras generadas para la imagen ID {image_id}: {len(masks) if masks else 0}")
            finally:
                # Eliminar el archivo temporal
                os.remove(temp_image_path)
                logger.info(f"Archivo temporal eliminado: {temp_image_path}")

            if not masks:
                logger.error("Error al generar las máscaras para la imagen generada.")
                raise HTTPException(
                    status_code=500,
                    detail="Error al generar las máscaras para la imagen generada."
                )

            # Procesar y guardar máscaras en la base de datos
            masks_data = self._process_and_save_masks(masks, image_id)
            logger.info(f"Mascaras guardadas para la imagen ID {image_id}: {masks_data}")

            return masks_data

        except HTTPException as he:
            raise he  # Re-raise HTTPException to be handled by FastAPI
        except Exception as e:
            logger.exception("Error en generate_masks_for_image.")
            raise HTTPException(
                status_code=500,
                detail="Error interno al generar máscaras para la imagen."
            )
