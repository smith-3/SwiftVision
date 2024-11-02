from typing import List
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from requests import Session
from app.ModelsAI import ModelsAI  # Importa solo la clase
from database import schemas
from database.database import get_db

# Initialize FastAPI application
# Inicialización de la aplicación FastAPI con metadatos
app = FastAPI(
    title="SWIFT VISION",
    version="1.0.0",
    description=(
        "API para la edición de imágenes basada en Segment Anything y Stable Diffusion, "
        "permitiendo edición avanzada mediante inteligencia artificial."
    )
)

# Configure CORS to allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Also allow OPTIONS
    allow_headers=["*"],
)

# Global variable to hold the ModelsAI object
modelsAI = None

@app.on_event("startup")
async def startup_event():
    """
    Inicializa los modelos de IA cuando se inicia la aplicación.
    """
    global modelsAI
    # Obtener la sesión de la base de datos
    db = next(get_db())  # Obtiene la sesión de la base de datos
    modelsAI = ModelsAI(db)  # Pasa la sesión al constructor de ModelsAI
    print("ModelsAI initialized")

@app.post("/login", response_model=schemas.User, tags=["Authentication"])
def login(user_credentials: schemas.UserCreate):
    """
    Autentica un usuario mediante su nombre de usuario y contraseña.

    - **username**: Nombre de usuario.
    - **password**: Contraseña del usuario.
    """
    user = modelsAI.crud.authenticate_user(
        user_credentials.username, user_credentials.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nombre de usuario o contraseña inválidos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@app.post("/register", response_model=schemas.User, tags=["Authentication"])
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Registra un nuevo usuario.

    - **username**: Nombre de usuario único.
    - **password**: Contraseña del usuario.
    """
    db_user = modelsAI.crud.get_user_by_username(username=user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El nombre de usuario ya está registrado."
        )
    return modelsAI.crud.create_user(user=user)

@app.get(
    "/projects/{project_id}/images/", response_model=List[schemas.Image], tags=["Image"]
)
def read_images(project_id: int):
    """
    Devuelve las imágenes de un proyecto específico.

    - **project_id**: ID del proyecto.
    """
    return modelsAI.crud.get_images_for_project(project_id=project_id)

@app.get(
    "/users/{user_id}/projects/", response_model=List[schemas.Project], tags=["Project"]
)
def read_projects(user_id: int):
    # Verificar si el usuario existe
    user = modelsAI.crud.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado."
        )
    
    # Obtener y devolver los proyectos del usuario
    projects = modelsAI.crud.get_projects(user_id=user_id)
    print(f"Proyectos encontrados para user_id={user_id}: {projects}")
    
    return projects  # Devuelve una lista vacía si no hay proyectos


@app.get(
    "/images/{image_id}/masks/",
    response_model=List[schemas.Mask],
    tags=["Mask"],
)
def read_masks(image_id: int):
    """
    Devuelve las anotaciones de una imagen específica.

    - **image_id**: ID de la imagen.
    """
    return modelsAI.crud.get_masks_for_image(image_id=image_id)

@app.post("/masks", tags=["Image"])
async def procesar_masks(user_id: int, project_name: str, file: UploadFile = File(...)):
    """
    Sube una imagen y genera máscaras para la misma.

    - **user_id**: ID del usuario.
    - **project_name**: Nombre del proyecto.
    - **file**: Archivo de imagen subido.
    """
    return modelsAI.process_image_and_generate_masks(
        user_id=user_id, project_name=project_name, file=file
    )


# @app.post("/masks_points", tags=['Image'])
# async def procesar_masks(
#     points: str = Form(...),
#     labels: str = Form(...)
# ):
#     try:
#         points = points.replace('(', '[').replace(')', ']')

#         # Imprimir para depuración
#         print(points)
#         print(labels)

#         # Convertir JSON a listas de Python
#         points_data = json.loads(points)
#         labels_data = json.loads(labels)

#         # Convertir listas a numpy arrays
#         points_array = np.array(points_data)
#         labels_array = np.array(labels_data)

#         # Llamar a la función con los numpy arrays
#         masks = segmentation.predict_mask_with_points(points_array, labels_array)
#         for mask in masks:
#             mask_data ={"segmentation": mask}
#             segmentation.masks_data.append(mask_data)
#         masks_data = []
#         for mask in masks:
#             if isinstance(mask, np.ndarray):
#                 area = int(np.sum(mask))  # Convertir el área a tipo nativo de Python
#                 size = mask.shape[::-1]
#                 encoded_matrix = run_length_encode_matrix(mask)
#                 mask_efficient = compress_encoded_matrix(encoded_matrix)
#                 segmentation_data = {
#                     "counts": mask_efficient,  # No lo tienes, así que lo dejamos vacío
#                     "size": size  # Tamaño de la máscara
#                 }
#                 masks_data.append({
#                     "segmentation": segmentation_data,
#                     "bbox": [],  # Cambiado a lista vacía
#                     "area": area,  # Área calculada de la máscara
#                     "predictedIou": 0,  # Cambiado a 0
#                     "stabilityScore": 0,  # Cambiado a 0
#                     "cropBox": [],  # Cambiado a lista vacía
#                     "pointCoords": []  # Cambiado a lista vacía
#                 })

#         # Retornar los datos de las máscaras como JSON
#         return JSONResponse(content={"masks": masks_data})

#     except Exception as e:
#         print(f"Error en procesar_imagen: {e}")  # Imprimir error en consola
#         return JSONResponse(status_code=500, content={"error": str(e)})

# Crear un nuevo proyecto
@app.post("/projects/", response_model=schemas.Project, tags=["Project"])
def create_project(project: schemas.ProjectCreate, user_id: int, db: Session = Depends(get_db)):
    """
    Crea un nuevo proyecto para un usuario específico.

    - **name**: Nombre del proyecto.
    - **user_id**: ID del usuario propietario.
    """
    return modelsAI.crud.create_project(project=project, user_id=user_id)

# Actualizar el nombre de un proyecto
@app.put("/projects/{project_id}", response_model=schemas.Project, tags=["Project"])
def update_project(project_id: int, project: schemas.ProjectUpdate, db: Session = Depends(get_db)):
    """
    Actualiza el nombre de un proyecto.

    - **name**: Nuevo nombre del proyecto.
    """
    db_project = modelsAI.crud.update_project(project_id=project_id, project_update=project)
    if not db_project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    return db_project

# Eliminar un proyecto
@app.delete("/projects/{project_id}", tags=["Project"])
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """
    Elimina un proyecto por su ID.
    """
    success = modelsAI.crud.delete_project(project_id=project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    return {"detail": "Proyecto eliminado exitosamente"}