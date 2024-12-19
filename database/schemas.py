from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import List, Union

class MaskBase(BaseModel):
    counts: List[List[Union[List[List[int]], int]]]  # Counts como lista serializable
    size: List[int]                      # [width, height]
    bbox: List[int]                      # [x, y, width, height]
    point_coords: List[List[float]]      # Lista de [x, y]

class MaskCreate(BaseModel):
    counts: List[List[Union[List[List[int]], int]]]  # Counts como lista serializable
    size: List[int]                                  # [width, height]
    bbox: List[float]                                # Lista de floats
    point_coords: List[List[float]]                  # Coordenadas en listas de floats

class Mask(MaskBase):
    id: int
    image_id: int
    created_at: datetime

    class Config:
        from_attributes = True
        orm_mode = True  # Habilita la conversión automática desde objetos ORM


class ImageBase(BaseModel):
    base_image: bytes

class ImageCreate(ImageBase):
    pass

class ImageBaseNoBinary(BaseModel):
    id: int
    project_id: int
    created_at: datetime
    masks: List[Mask] = []

    class Config:
        orm_mode = True

class ProjectBase(BaseModel):
    name: str

class ProjectCreate(ProjectBase):
    pass


class ImageBaseNoBinary(BaseModel):
    id: int
    project_id: int
    created_at: datetime

    class Config:
        from_attributes = True
        

class Project(ProjectBase):
    id: int
    user_id: int
    created_at: datetime
    images: List[ImageBaseNoBinary] = []

    class Config:
        from_attributes = True

class UserBase(BaseModel):
    email: str  # Ahora se usa email en lugar de username
    username: Optional[str] = None  # Mantener opcional por compatibilidad

class Login(BaseModel):
    email: str  # Ahora se usa email en lugar de username
    password: str


class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    projects: List[Project] = []

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    """
    Define los campos opcionales que pueden ser actualizados en un usuario.
    """
    username: Optional[str] = None
    password: Optional[str] = None

    class Config:
        from_attributes = True

class ProjectUpdate(BaseModel):
    """
    Define los campos opcionales que pueden ser actualizados en un proyecto.
    """
    name: Optional[str] = None

    class Config:
        from_attributes = True

class ImageUpdate(BaseModel):
    """
    Define los campos opcionales que pueden ser actualizados en una imagen.
    """
    base_image: Optional[bytes] = None

    class Config:
        from_attributes = True
class MaskUpdate(BaseModel):
    """
    Define los campos opcionales que pueden ser actualizados en una anotación.
    """
    counts: Optional[str] = None
    size: Optional[str] = None
    bbox: Optional[str] = None
    point_coords: Optional[str] = None

    class Config:
        from_attributes = True

