from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from base64 import b64encode

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

class Image(BaseModel):
    id: int
    project_id: int
    created_at: datetime
    base_image: str  # Base64 codificado
    masks: List[Mask] = []

    class Config:
        orm_mode = True

    @staticmethod
    def encode_image_to_base64(binary_data: bytes) -> str:
        """Convierte los datos binarios a una cadena Base64."""
        return b64encode(binary_data).decode('utf-8')

    @classmethod
    def from_orm(cls, obj):
        """Sobrescribe la conversión para codificar 'base_image' a Base64."""
        obj_dict = super().from_orm(obj).dict()
        if obj.base_image:
            obj_dict["base_image"] = cls.encode_image_to_base64(obj.base_image)
        return cls.parse_obj(obj_dict)


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

