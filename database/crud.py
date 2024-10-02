from sqlalchemy.orm import Session

from . import models
from . import schemas
import bcrypt

# CRUD para User
# Encripta la contraseña antes de almacenar el usuario en la base de datos
def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())  # Hash de la contraseña
    db_user = models.User(username=user.username, password=hashed_password.decode('utf-8'))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def login(db: Session, username: str, password: str):
    user = authenticate_user(db, username, password)
    if not user:
        raise ValueError("Invalid username or password")
    return user


# Función para autenticar a un usuario comparando la contraseña con la almacenada
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        return None
    if not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return None
    return user

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.User).offset(skip).limit(limit).all()

# database/crud.py
def get_user_by_username(db, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

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
