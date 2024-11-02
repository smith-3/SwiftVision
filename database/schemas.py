from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class MaskBase(BaseModel):
    counts: str
    size: str
    bbox: Optional[str]
    point_coords: Optional[str]

class MaskCreate(MaskBase):
    pass

class Mask(MaskBase):
    id: int
    image_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ImageBase(BaseModel):
    base_image: bytes

class ImageCreate(ImageBase):
    pass

class Image(ImageBase):
    id: int
    project_id: int
    created_at: datetime
    Masks: List[Mask] = []

    class Config:
        from_attributes = True

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
    username: str

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

