import tempfile
from fastapi import UploadFile, HTTPException, File, Depends
from sqlalchemy.orm import Session
from app.Lama import Lama
from app.Segment import SAM
from app.StableDiffusion import StableDiffusion
from utils.compress_descompress import compress_encoded_matrix, decompress_encoded_matrix, run_length_encode_matrix
from database.crud import CRUDOperations
from database.schemas import ProjectCreate, ImageCreate, MaskCreate
from fastapi.responses import JSONResponse
import numpy as np
import os
import PIL.Image as Image
import io

from utils.utils import dilate_mask

class ModelsAI:
    def __init__(self, db: Session):
        self.crud = CRUDOperations(db)
        self.sam = SAM()
        self.lama = Lama()
        self.stableDiffusion = StableDiffusion()

    def generate_image(self, project_id: int, image_id: int, mask_id: int, promt: str):
        try:
            # Obtener la imagen y la máscara de la base de datos
            db_image = self.crud.get_image(image_id)
            db_mask = self.crud.get_mask(mask_id)
            db_project = self.crud.get_project(project_id)

            if not db_image or not db_mask or not db_project:
                raise HTTPException(status_code=404, detail="Imagen, máscara o proyecto no encontrado.")

            # Cargar la imagen desde los datos binarios de la base de datos
            img_array = np.frombuffer(db_image.base_image, dtype=np.uint8)
            img = Image.open(io.BytesIO(img_array)).convert("RGB")  # Convertir a RGB
            img = np.array(img)

            # Validar y configurar img
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise HTTPException(status_code=400, detail="La imagen no tiene el formato esperado (RGB).")

            # Decodificar la máscara usando la función decompress_encoded_matrix
            mask_counts = eval(db_mask.counts)  # Convertir el string a la matriz comprimida
            mask_size = eval(db_mask.size)  # Convertir el tamaño a una tupla (alto, ancho)
            mask = decompress_encoded_matrix(mask_counts, (mask_size[1], mask_size[0]))

            mask = dilate_mask(mask, 30)

            # Validar y configurar mask
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            if len(mask.shape) != 2:
                raise HTTPException(status_code=400, detail="La máscara no tiene el formato esperado (2D).")
            if np.max(mask) == 1:
                mask = mask * 255  # Escalar valores de binario a 0-255

            # Asegurarse de que las dimensiones de la imagen y la máscara coincidan
            if img.shape[:2] != mask.shape:
                raise HTTPException(status_code=400, detail="La imagen y la máscara tienen dimensiones diferentes.")

            # Usar Lama para realizar el inpainting
            self.save_images(img, mask)

            img_inpainted = self.stableDiffusion.fill_img_with_sd(img,mask,promt)
            # img_inpainted = self.lama.inpaint(img, mask)

            # Asegurarse de que la imagen esté en formato uint8
            img_inpainted = img_inpainted.astype(np.uint8)
            
            # Convertir np.array a imagen PIL
            img_pil = Image.fromarray(img_inpainted)
            
            # Crear un buffer de memoria
            byte_io = io.BytesIO()
            
            # Guardar la imagen en el buffer como PNG (puedes elegir otros formatos si lo prefieres)
            img_pil.save(byte_io, format='PNG')
            
            # Obtener los bytes del buffer
            img_bytes = byte_io.getvalue()
            # Guardar la imagen inpainted como nueva imagen en la base de datos
            
            inpainted_image_data = ImageCreate(base_image=img_bytes)
            inpainted_image = self.crud.create_image(image=inpainted_image_data, project_id=db_project.id)

            # Guardar la imagen inpainted localmente
            output_dir = "./inpainted_images"  # Directorio local donde se guardará la imagen
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Definir la ruta de la imagen
            output_image_path = os.path.join(output_dir, f"inpainted_image_{db_image.id}.png")

            # Guardar la imagen inpainted en el directorio local
            img_inpainted_pil = Image.fromarray(img_inpainted)
            img_inpainted_pil.save(output_image_path)


            # Retornar el ID de la nueva imagen inpainted
            return {"inpainted_image_id": inpainted_image.id}

        except HTTPException as he:
            raise he  # Re-raise HTTPException para ser manejado por FastAPI
        except Exception as e:
            print(f"Error en process_inpainting_with_mask: {e}")  # Imprimir error en consola
            raise HTTPException(status_code=500, detail=str(e))
        
    def generate_background(self, project_id: int, image_id: int, mask_id: int, promt: str):
        try:
            # Obtener la imagen y la máscara de la base de datos
            db_image = self.crud.get_image(image_id)
            db_mask = self.crud.get_mask(mask_id)
            db_project = self.crud.get_project(project_id)

            if not db_image or not db_mask or not db_project:
                raise HTTPException(status_code=404, detail="Imagen, máscara o proyecto no encontrado.")

            # Cargar la imagen desde los datos binarios de la base de datos
            img_array = np.frombuffer(db_image.base_image, dtype=np.uint8)
            img = Image.open(io.BytesIO(img_array)).convert("RGB")  # Convertir a RGB
            img = np.array(img)

            # Validar y configurar img
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise HTTPException(status_code=400, detail="La imagen no tiene el formato esperado (RGB).")

            # Decodificar la máscara usando la función decompress_encoded_matrix
            mask_counts = eval(db_mask.counts)  # Convertir el string a la matriz comprimida
            mask_size = eval(db_mask.size)  # Convertir el tamaño a una tupla (alto, ancho)
            mask = decompress_encoded_matrix(mask_counts, (mask_size[1], mask_size[0]))

            mask = dilate_mask(mask, 30)

            # Validar y configurar mask
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            if len(mask.shape) != 2:
                raise HTTPException(status_code=400, detail="La máscara no tiene el formato esperado (2D).")
            if np.max(mask) == 1:
                mask = mask * 255  # Escalar valores de binario a 0-255

            # Asegurarse de que las dimensiones de la imagen y la máscara coincidan
            if img.shape[:2] != mask.shape:
                raise HTTPException(status_code=400, detail="La imagen y la máscara tienen dimensiones diferentes.")

            # Usar Lama para realizar el inpainting
            self.save_images(img, mask)

            img_inpainted = self.stableDiffusion.replace_img_with_sd(img,mask,promt)
            # img_inpainted = self.lama.inpaint(img, mask)

            # Asegurarse de que la imagen esté en formato uint8
            img_inpainted = img_inpainted.astype(np.uint8)
            
            # Convertir np.array a imagen PIL
            img_pil = Image.fromarray(img_inpainted)
            
            # Crear un buffer de memoria
            byte_io = io.BytesIO()
            
            # Guardar la imagen en el buffer como PNG (puedes elegir otros formatos si lo prefieres)
            img_pil.save(byte_io, format='PNG')
            
            # Obtener los bytes del buffer
            img_bytes = byte_io.getvalue()
            # Guardar la imagen inpainted como nueva imagen en la base de datos
            
            inpainted_image_data = ImageCreate(base_image=img_bytes)
            inpainted_image = self.crud.create_image(image=inpainted_image_data, project_id=db_project.id)

            # Guardar la imagen inpainted localmente
            output_dir = "./inpainted_images"  # Directorio local donde se guardará la imagen
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Definir la ruta de la imagen
            output_image_path = os.path.join(output_dir, f"inpainted_image_{db_image.id}.png")

            # Guardar la imagen inpainted en el directorio local
            img_inpainted_pil = Image.fromarray(img_inpainted)
            img_inpainted_pil.save(output_image_path)


            # Retornar el ID de la nueva imagen inpainted
            return {"inpainted_image_id": inpainted_image.id}

        except HTTPException as he:
            raise he  # Re-raise HTTPException para ser manejado por FastAPI
        except Exception as e:
            print(f"Error en process_inpainting_with_mask: {e}")  # Imprimir error en consola
            raise HTTPException(status_code=500, detail=str(e))
        
    def process_inpainting_with_mask(self, project_id: int, image_id: int, mask_id: int):
        try:
            # Obtener la imagen y la máscara de la base de datos
            db_image = self.crud.get_image(image_id)
            db_mask = self.crud.get_mask(mask_id)
            db_project = self.crud.get_project(project_id)

            if not db_image or not db_mask or not db_project:
                raise HTTPException(status_code=404, detail="Imagen, máscara o proyecto no encontrado.")

            # Cargar la imagen desde los datos binarios de la base de datos
            img_array = np.frombuffer(db_image.base_image, dtype=np.uint8)
            img = Image.open(io.BytesIO(img_array)).convert("RGB")  # Convertir a RGB
            img = np.array(img)

            # Validar y configurar img
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise HTTPException(status_code=400, detail="La imagen no tiene el formato esperado (RGB).")

            # Decodificar la máscara usando la función decompress_encoded_matrix
            mask_counts = eval(db_mask.counts)  # Convertir el string a la matriz comprimida
            mask_size = eval(db_mask.size)  # Convertir el tamaño a una tupla (alto, ancho)
            mask = decompress_encoded_matrix(mask_counts, (mask_size[1], mask_size[0]))

            mask = dilate_mask(mask, 30)

            # Validar y configurar mask
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            if len(mask.shape) != 2:
                raise HTTPException(status_code=400, detail="La máscara no tiene el formato esperado (2D).")
            if np.max(mask) == 1:
                mask = mask * 255  # Escalar valores de binario a 0-255

            # Asegurarse de que las dimensiones de la imagen y la máscara coincidan
            if img.shape[:2] != mask.shape:
                raise HTTPException(status_code=400, detail="La imagen y la máscara tienen dimensiones diferentes.")

            # Usar Lama para realizar el inpainting
            self.save_images(img, mask)
            img_inpainted = self.lama.inpaint(img, mask)

            # Asegurarse de que la imagen esté en formato uint8
            img_inpainted = img_inpainted.astype(np.uint8)
            
            # Convertir np.array a imagen PIL
            img_pil = Image.fromarray(img_inpainted)
            
            # Crear un buffer de memoria
            byte_io = io.BytesIO()
            
            # Guardar la imagen en el buffer como PNG (puedes elegir otros formatos si lo prefieres)
            img_pil.save(byte_io, format='PNG')
            
            # Obtener los bytes del buffer
            img_bytes = byte_io.getvalue()
            # Guardar la imagen inpainted como nueva imagen en la base de datos
            
            inpainted_image_data = ImageCreate(base_image=img_bytes)
            inpainted_image = self.crud.create_image(image=inpainted_image_data, project_id=db_project.id)

            # Guardar la imagen inpainted localmente
            output_dir = "./inpainted_images"  # Directorio local donde se guardará la imagen
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Definir la ruta de la imagen
            output_image_path = os.path.join(output_dir, f"inpainted_image_{db_image.id}.png")

            # Guardar la imagen inpainted en el directorio local
            img_inpainted_pil = Image.fromarray(img_inpainted)
            img_inpainted_pil.save(output_image_path)


            # Retornar el ID de la nueva imagen inpainted
            return {"inpainted_image_id": inpainted_image.id}

        except HTTPException as he:
            raise he  # Re-raise HTTPException para ser manejado por FastAPI
        except Exception as e:
            print(f"Error en process_inpainting_with_mask: {e}")  # Imprimir error en consola
            raise HTTPException(status_code=500, detail=str(e))

    def save_images(self, img: np.array, mask: np.array):
        try:
            # Asegurarse de que las imágenes sean arrays de tipo uint8
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            # Guardar las imágenes localmente
            output_dir = "./saved_images"  # Directorio local donde se guardarán las imágenes
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Guardar la imagen 'img'
            img_pil = Image.fromarray(img)
            img_filename = f"image_1.png"
            img_path = os.path.join(output_dir, img_filename)
            img_pil.save(img_path)

            # Guardar la máscara 'mask'
            mask_pil = Image.fromarray(mask)
            mask_filename = f"mask_1.png"
            mask_path = os.path.join(output_dir, mask_filename)
            mask_pil.save(mask_path)
        except Exception as e:
            print(f"Error en save_images: {e}")
            raise HTTPException(status_code=500, detail="Error al guardar las imágenes")

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
                        "size": str(segmentation_data.shape[::-1])   # Convertir a string para almacenarlo
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
                mask_data = self.crud.create_mask(mask=mask_data, image_id=db_image.id)
                masks_data.append({
                    "id": mask_data.id,  # O el atributo que necesites
                    "counts": mask_data.counts,
                    "size": mask_data.size,
                    "bbox": mask_data.bbox,
                    "point_coords": mask_data.point_coords
                })
            print("Masks Data:", masks_data)  # Imprimir para depuración

            return JSONResponse(content={"id": db_project.id})

        except HTTPException as he:
            raise he  # Re-raise HTTPException to be handled by FastAPI
        except Exception as e:
            print(f"Error en process_image_and_generate_masks: {e}")  # Imprimir error en consola
            raise HTTPException(status_code=500, detail=str(e))