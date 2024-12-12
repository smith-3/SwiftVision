from typing import List
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, status, APIRouter, Form
from fastapi.middleware.cors import CORSMiddleware
from requests import Session
from app.ModelsAI import ModelsAI  # Importa solo la clase
from database import schemas
from database.database import get_db
from fastapi.responses import StreamingResponse
import io

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
def login(user_credentials: schemas.Login):
    """
    Autentica un usuario mediante su correo y contraseña.

    - **email**: Correo del usuario.
    - **password**: Contraseña del usuario.
    """
    print(user_credentials)
    user = modelsAI.crud.authenticate_user_by_email(
        email=user_credentials.email, password=user_credentials.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña inválidos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@app.post("/register", response_model=schemas.User, tags=["Authentication"])
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Registra un nuevo usuario.

    - **email**: Correo único del usuario.
    - **password**: Contraseña del usuario.
    """
    print(user)
    db_user = modelsAI.crud.get_user_by_email(email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El correo ya está registrado."
        )
    return modelsAI.crud.create_user(user=user)

@app.get(
    "/images/{project_id}/",
    response_model=List[schemas.ImageBaseNoBinary],
    tags=["Image"]
)
def read_images(project_id: int):
    """
    Devuelve las imágenes de un proyecto específico 

    - **project_id**: ID del proyecto.
    """
    images = modelsAI.crud.get_images_for_project(project_id=project_id)
    if not images:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron imágenes para el proyecto."
        )
    return images

@app.get(
    "/projects/{user_id}/", response_model=List[schemas.Project], tags=["Project"]
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

# Crear un nuevo proyecto
@app.post("/projects", tags=["Project"])
async def procesar_masks(
    user_id: int = Form(...), 
    project_name: str = Form(...), 
    file: UploadFile = File(...)
):
    """
    Sube una imagen y genera máscaras para la misma.

    - **user_id**: ID del usuario.
    - **project_name**: Nombre del proyecto.
    - **file**: Archivo de imagen subido.
    """
    try:
        # Procesar el archivo y generar máscaras
        print(user_id, project_name)
        result = modelsAI.process_image_and_generate_masks(
            user_id=user_id, 
            project_name=project_name, 
            file=file
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/process_inpainting_with_mask", tags=["Edit"])
def process_inpainting(project_id: int, image_id: int, mask_id: int, db: Session = Depends(get_db)):
    result = modelsAI.process_inpainting_with_mask(project_id, image_id, mask_id)
    return result

@app.get("/generate_image", tags=["Edit"])
def generate_image(project_id: int, image_id: int, mask_id: int,  promt: str, db: Session = Depends(get_db)):
    result = modelsAI.generate_image(project_id, image_id, mask_id,promt)
    return result

@app.get("/generate_image_backgraund", tags=["Edit"])
def generate_image(project_id: int, image_id: int, mask_id: int,  promt: str, db: Session = Depends(get_db)):
    result = modelsAI.generate_background(project_id, image_id, mask_id,promt)
    return result

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


@app.get("/users/{user_id}", response_model=schemas.User, tags=["User"])
def get_user(user_id: int, db: Session = Depends(get_db)):
    """
    Devuelve los datos de un usuario específico.

    - **user_id**: ID del usuario.
    """
    user = modelsAI.crud.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado."
        )
    return user

@app.get("/users", response_model=List[schemas.User], tags=["User"])
def get_all_users(db: Session = Depends(get_db)):
    """
    Devuelve una lista de todos los usuarios registrados.

    - **Respuesta**: Lista de objetos de usuario.
    """
    users = modelsAI.crud.get_users()
    if not users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontraron usuarios."
        )
    return users

@app.get("/images/{image_id}/download", tags=["Image"])
def download_image(image_id: int):
    """
    Descarga la imagen binaria por su ID.
    """
    image = modelsAI.crud.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
    return StreamingResponse(io.BytesIO(image.base_image), media_type="image/png")



@app.get("/images/{image_id}/masks/", response_model=List[schemas.Mask], tags=["Mask"])
def read_masks(image_id: int, db: Session = Depends(get_db)):
    """
    Devuelve todas las máscaras asociadas a una imagen específica.

    - **image_id**: ID de la imagen.
    """
    masks = modelsAI.crud.get_masks_for_image(image_id)
    if not masks:
        raise HTTPException(
            status_code=404, 
            detail="No se encontraron máscaras para esta imagen."
        )
    return masks

@app.delete("/images/{image_id}", tags=["Image"])
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """
    Elimina una imagen específica por su ID.
    """
    success = modelsAI.crud.delete_image(image_id=image_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Imagen no encontrada."
        )
    return {"detail": "Imagen eliminada exitosamente"}

@app.post("/masks_points", tags=["Mask"])
def upload_mask(
    project_id: int = Form(...),
    image_id: int = Form(...),
    counts: str = Form(...),
    size: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Recibe la máscara comprimida RLE (counts) y el tamaño (size),
    y la guarda en la base de datos asociada a la imagen y proyecto.
    """
    # Parsear el JSON recibido en `counts` (run-length encoding comprimido)
    try:
        compressed_rows = json.loads(counts)  # List[CompressedRow], p. ej.

        # size vendrá con un formato "(width, height)" o algo similar:
        # Ej: "(640, 480)"
        size_str = size.strip().replace("(", "").replace(")", "")
        width, height = [int(s) for s in size_str.split(",")]

        # En este ejemplo, guardamos directamente `counts` y `size` como string en la BD.
        # Podrías decodificarlo y almacenarlo con un formato diferente según tus necesidades.

        new_mask_data = MaskCreate(
            counts=counts,     # el string JSON comprimido
            size=size,         # "(width, height)"
            bbox="[]",         # si no tienes bounding box, usa "[]"
            point_coords="[]", # si no tienes coords, usa "[]"
            active=True        # o false, depende de tu lógica
        )

        # Creación de la máscara en la BD con tu capa de CRUD
        mask_db = ModelsAI(db).crud.create_mask(mask=new_mask_data, image_id=image_id)

        return {
            "detail": "Mask uploaded successfully",
            "mask_id": mask_db.id,
            "project_id": project_id,
            "image_id": image_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al procesar la máscara: {str(e)}"
        )