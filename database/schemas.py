from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class AnnotationBase(BaseModel):
    segmentation: str
    bbox: Optional[str]
    crop_box: Optional[str]
    point_coords: Optional[str]

class AnnotationCreate(AnnotationBase):
    pass

class Annotation(AnnotationBase):
    id: int
    image_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ImageBase(BaseModel):
    base_image: str

class ImageCreate(ImageBase):
    pass

class Image(ImageBase):
    id: int
    project_id: int
    created_at: datetime
    annotations: List[Annotation] = []

    class Config:
        from_attributes = True

class ProjectBase(BaseModel):
    name: str

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    user_id: int
    created_at: datetime
    images: List[Image] = []

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
