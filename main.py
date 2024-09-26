import json
from typing import Union
import os
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from classes.Segment import ImageSegmentation
from utils.compress_descompress import combine_boolean_matrices, compress_encoded_matrix, run_length_encode_matrix

# Initialize FastAPI application
app = FastAPI()
app.title = "IGORA"
app.version = "1.0.0"
app.description = "Todos los servicios de color en el salon se pueden realizar en IGORA"

# Configure CORS to allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Also allow OPTIONS
    allow_headers=["*"],
)

# Global variable to hold the ImageSegmentation object
segmentation = None

@app.on_event("startup")
async def startup_event():
    global segmentation, masks
    segmentation = ImageSegmentation()
    masks = None  # Inicializar masks
    print("ImageSegmentation initialized")

@app.get("/", tags=['Home'])
def main():
    return JSONResponse(content={"message": "Bienvenido a IGORA"})

@app.post("/masks", tags=['Image'])
async def procesar_masks(file: UploadFile = File(...)):
    try:
        image_path = f"./response/{file.filename}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail="Error al guardar el archivo.")
        segmentation.load_image(image_path)
        masks = segmentation.generate_masks()
        if masks is None:
            raise HTTPException(status_code=500, detail="Error al generar las máscaras.")
        masks_data = []
        for mask in masks:
            segmentation_data = mask.get("segmentation", None)
            if isinstance(segmentation_data, np.ndarray):
                counts = np.array(segmentation_data, dtype=bool)
                encoded_matrix = run_length_encode_matrix(counts)
                mask_efficient = compress_encoded_matrix(encoded_matrix)
                segmentation_data = {
                    "counts": mask_efficient,
                    "size": segmentation_data.shape[::-1]
                }
            else:
                segmentation_data = {
                    "counts": mask.get("counts", []),
                    "size": mask.get("size", "N/A")
                }
            masks_data.append({
                "segmentation": segmentation_data,
                "bbox": mask.get("bbox", "N/A"),
                "area": mask.get("area", "N/A"),
                "predictedIou": mask.get("predicted_iou", "N/A"),
                "stabilityScore": mask.get("stability_score", "N/A"),
                "cropBox": mask.get("crop_box", "N/A"),
                "pointCoords": mask.get("point_coords", "N/A")
            })
        return JSONResponse(content={"masks": masks_data})

    except Exception as e:
        print(f"Error en procesar_imagen: {e}")  # Print error to console
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/colorize", tags=['Image'])
async def colorize_image(
    masks_json: str = Form(...),    # Recibir el JSON con las máscaras
    rgb: str = Form(...)           # Recibir el valor RGB
):
    try:
        decoded_masks = []
        print(masks_json)
        try:
            masks = json.loads(masks_json)
            for indice in masks:
                mask = segmentation.masks_data[indice]
                segmentation_data = mask.get("segmentation", None)
                counts = np.array(segmentation_data, dtype=bool)
                decoded_masks.append(counts)
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Error en el formato de masks_json: {e}")
        try:
            print(rgb)
            rgb_values = tuple(map(int, rgb.split(',')))
        except ValueError:
            raise HTTPException(status_code=400, detail="El valor RGB no tiene el formato correcto.")
        combined_mask = combine_boolean_matrices(decoded_masks)
        print("Mascara combinada")
        refined_mask = segmentation.refine_hair_mask(combined_mask)
        print("Refinado")
        segmentation.hair_color(refined_mask, rgb_values, "./response/colored_image.png")
        print("Cambio de color listo")
        return FileResponse("./response/colored_image.png", media_type="image/png")
    except Exception as e:
        print(f"Error en colorize_image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/masks_points", tags=['Image'])
async def procesar_masks(
    points: str = Form(...),
    labels: str = Form(...)
):
    try:
        points = points.replace('(', '[').replace(')', ']')

        # Imprimir para depuración
        print(points)
        print(labels)

        # Convertir JSON a listas de Python
        points_data = json.loads(points)
        labels_data = json.loads(labels)

        # Convertir listas a numpy arrays
        points_array = np.array(points_data)
        labels_array = np.array(labels_data)

        # Llamar a la función con los numpy arrays
        masks = segmentation.predict_mask_with_points(points_array, labels_array)
        for mask in masks:
            mask_data ={"segmentation": mask}
            segmentation.masks_data.append(mask_data)
        masks_data = []
        for mask in masks:
            if isinstance(mask, np.ndarray):
                area = int(np.sum(mask))  # Convertir el área a tipo nativo de Python
                size = mask.shape[::-1]
                encoded_matrix = run_length_encode_matrix(mask)
                mask_efficient = compress_encoded_matrix(encoded_matrix)
                segmentation_data = {
                    "counts": mask_efficient,  # No lo tienes, así que lo dejamos vacío
                    "size": size  # Tamaño de la máscara
                }
                masks_data.append({
                    "segmentation": segmentation_data,
                    "bbox": [],  # Cambiado a lista vacía
                    "area": area,  # Área calculada de la máscara
                    "predictedIou": 0,  # Cambiado a 0
                    "stabilityScore": 0,  # Cambiado a 0
                    "cropBox": [],  # Cambiado a lista vacía
                    "pointCoords": []  # Cambiado a lista vacía
                })

        # Retornar los datos de las máscaras como JSON
        return JSONResponse(content={"masks": masks_data})

    except Exception as e:
        print(f"Error en procesar_imagen: {e}")  # Imprimir error en consola
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/colorize_hair", tags=['Image'])
async def colorize_image(
    file: UploadFile = File(...),  # Recibir el archivo de imagen
    rgb: str = Form(...)           # Recibir el valor RGB
):
    try:
        # Imprimir los valores recibidos
        print(f"Valor RGB recibido: {rgb}")
        print(f"Nombre del archivo recibido: {file.filename}")

        # Validar y convertir el valor RGB
        try:
            rgb_values = tuple(map(float, rgb.split(',')))
            print(f"Valores RGB convertidos a tupla: {rgb_values}")
        except ValueError:
            print("Error: El valor RGB no tiene el formato correcto.")
            raise HTTPException(status_code=400, detail="El valor RGB no tiene el formato correcto.")

        # Guardar la imagen
        image_path = f"./response/{file.filename}"
        print(f"Ruta donde se guardará la imagen: {image_path}")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())
        print("Archivo de imagen guardado exitosamente.")

        # Verificar si la imagen fue guardada correctamente
        if not os.path.exists(image_path):
            print(f"Error: No se pudo guardar el archivo en {image_path}")
            raise HTTPException(status_code=400, detail="Error al guardar el archivo.")

        # Cargar la imagen en el sistema de segmentación
        segmentation.load_image(image_path)
        print("Imagen cargada para segmentación.")

        # Generar las máscaras
        masks = segmentation.generate_masks()
        print("Máscaras generadas.")
        if masks is None:
            print("Error: No se pudieron generar las máscaras.")
            raise HTTPException(status_code=500, detail="Error al generar las máscaras.")
        
        sorted_anns = sorted(masks, key=lambda x: x['area'], reverse=False)
        masks = sorted_anns
        # Encontrar la máscara del cabello
        hair = None
        for mask in sorted_anns:
            if getMask(mask['segmentation']):
                hair = mask['segmentation']
                break  # Salir del bucle si se encuentra la máscara
        if hair is None:
            print("Error: No se pudo encontrar la máscara de cabello.")
            raise HTTPException(status_code=404, detail="No se encontró la máscara de cabello.")
        else:
            print("Máscara de cabello obtenida.")

        # Refinar la máscara del cabello
        refined_mask = segmentation.refine_hair_mask(hair)
        print("Máscara de cabello refinada.")

        # Cambiar el color del cabello
        segmentation.hair_color(refined_mask, rgb_values, "./response/colored_image.png")
        print("Cambio de color de cabello aplicado exitosamente.")

        # Devolver la imagen final
        return FileResponse("./response/colored_image.png", media_type="image/png")

    except HTTPException as http_exc:
        # Manejo específico para HTTPException
        print(f"HTTPException: {http_exc.detail}")
        return JSONResponse(status_code=http_exc.status_code, content={"error": http_exc.detail})
    except Exception as e:
        # Manejo general de excepciones
        print(f"Error en colorize_image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

        return JSONResponse(status_code=500, content={"error": str(e)})