import json
from typing import List, Union
import os
from fastapi import Depends, FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from requests import Session
from app.Segment import ImageSegmentation
from database import crud, schemas
from database.database import get_db
from utils.compress_descompress import combine_boolean_matrices, compress_encoded_matrix, decompress_encoded_matrix, getMask, run_length_encode_matrix

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

@app.post("/login", response_model=schemas.User)
def login(user_credentials: schemas.UserCreate, db: Session = Depends(get_db)):
    user = crud.authenticate_user(db, user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Rutas para User
@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    return crud.create_user(db=db, user=user)

# @app.get("/users/", response_model=List[schemas.User])
# def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
#     users = crud.get_users(db, skip=skip, limit=limit)
#     return users

# @app.get("/users/{user_id}", response_model=schemas.User)
# def read_user(user_id: int, db: Session = Depends(get_db)):
#     db_user = crud.get_user(db, user_id=user_id)
#     if db_user is None:
#         raise HTTPException(status_code=404, detail="User not found")
#     return db_user

# Rutas para Project
# @app.post("/users/{user_id}/projects/", response_model=schemas.Project)
# def create_project_for_user(user_id: int, project: schemas.ProjectCreate, db: Session = Depends(get_db)):
#     return crud.create_project(db=db, project=project, user_id=user_id)

@app.get("/users/{user_id}/projects/", response_model=List[schemas.Project])
def read_projects(user_id: int, db: Session = Depends(get_db)):
    return crud.get_projects(db, user_id=user_id)

# Rutas para Image
# @app.post("/projects/{project_id}/images/", response_model=schemas.Image)
# def create_image_for_project(project_id: int, image: schemas.ImageCreate, db: Session = Depends(get_db)):
#     return crud.create_image(db=db, image=image, project_id=project_id)

@app.get("/projects/{project_id}/images/", response_model=List[schemas.Image])
def read_images(project_id: int, db: Session = Depends(get_db)):
    # Aquí podrías implementar un filtro si es necesario
    return crud.get_images_for_project(db, project_id=project_id)

# Rutas para Annotation
# @app.post("/images/{image_id}/annotations/", response_model=schemas.Annotation)
# def create_annotation_for_image(image_id: int, annotation: schemas.AnnotationCreate, db: Session = Depends(get_db)):
#     return crud.create_annotation(db=db, annotation=annotation, image_id=image_id)

@app.get("/images/{image_id}/annotations/", response_model=List[schemas.Annotation])
def read_annotations(image_id: int, db: Session = Depends(get_db)):
    return crud.get_annotations_for_image(db, image_id=image_id)


@app.post("/masks", tags=['Image'])
async def procesar_masks(file: UploadFile = File(...)):
    try:
        image_path = f"./output/{file.filename}"
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
            masks_data.append(segmentation_data)
        return JSONResponse(content={"masks": masks_data})

    except Exception as e:
        print(f"Error en procesar_imagen: {e}")  # Print error to console
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

@app.post("/lama", tags=['Image'])
async def procesar_masks(
    file: UploadFile = File(...),
    mask: str = Form(...),
    size: str = Form(...)
):
    try:
        image_path = f"./output/{file.filename}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail="Error al guardar el archivo.")
        mask_process = decompress_encoded_matrix(mask, size)
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
            masks_data.append(segmentation_data)
        return JSONResponse(content={"masks": masks_data})

    except Exception as e:
        print(f"Error en procesar_imagen: {e}")  # Print error to console
        return JSONResponse(status_code=500, content={"error": str(e)})
