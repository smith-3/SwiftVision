from sqlalchemy.orm import Session

from . import models
from . import schemas

# CRUD para User
def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(username=user.username, password=user.password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.User).offset(skip).limit(limit).all()

# CRUD para Project
def create_project(db: Session, project: schemas.ProjectCreate, user_id: int):
    db_project = models.Project(**project.dict(), user_id=user_id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

def get_projects(db: Session, user_id: int):
    return db.query(models.Project).filter(models.Project.user_id == user_id).all()

# CRUD para Image
def create_image(db: Session, image: schemas.ImageCreate, project_id: int):
    db_image = models.Image(**image.dict(), project_id=project_id)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

# CRUD para Annotation
def create_annotation(db: Session, annotation: schemas.AnnotationCreate, image_id: int):
    db_annotation = models.Annotation(**annotation.dict(), image_id=image_id)
    db.add(db_annotation)
    db.commit()
    db.refresh(db_annotation)
    return db_annotation
