from sqlalchemy.orm import Session
from . import models
from . import schemas
import bcrypt
from sqlalchemy.orm import joinedload

class CRUDOperations:
    def __init__(self, db: Session):
        self.db = db

    # User CRUD
    def create_user(self, user: schemas.UserCreate):
        hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
        db_user = models.User(username=user.username,email=user.email, password=hashed_password.decode('utf-8'))
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def authenticate_user_by_email(self, email: str, password: str):
        user = self.get_user_by_email(email)
        if user is None or not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            return None
        return user

    def authenticate_user(self, username: str, password: str):
        user = self.db.query(models.User).filter(models.User.username == username).first()
        if user is None or not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            return None
        return user

    def get_user(self, user_id: int):
        return self.db.query(models.User).filter(models.User.id == user_id).first()

    def get_user_by_username(self, username: str):
        return self.db.query(models.User).filter(models.User.username == username).first()

    def get_users(self, skip: int = 0, limit: int = 10):
        return self.db.query(models.User).offset(skip).limit(limit).all()

    def update_user(self, user_id: int, user_update: schemas.UserUpdate):
        db_user = self.get_user(user_id)
        if db_user is None:
            return None
        for key, value in user_update.dict(exclude_unset=True).items():
            setattr(db_user, key, value)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def delete_user(self, user_id: int):
        db_user = self.get_user(user_id)
        if db_user is None:
            return None
        self.db.delete(db_user)
        self.db.commit()
        return db_user

    # Project CRUD
    def create_project(self, project: schemas.ProjectCreate, user_id: int):
        db_project = models.Project(**project.dict(), user_id=user_id)
        self.db.add(db_project)
        self.db.commit()
        self.db.refresh(db_project)
        return db_project

    def get_projects(self, user_id: int):
        return self.db.query(models.Project).filter(models.Project.user_id == user_id).all()

    def get_project(self, project_id: int):
        return self.db.query(models.Project).filter(models.Project.id == project_id).first()

    def update_project(self, project_id: int, project_update: schemas.ProjectUpdate):
        db_project = self.get_project(project_id)
        if db_project is None:
            return None
        for key, value in project_update.dict(exclude_unset=True).items():
            setattr(db_project, key, value)
        self.db.commit()
        self.db.refresh(db_project)
        return db_project

    def delete_project(self, project_id: int):
        db_project = self.get_project(project_id)
        if db_project is None:
            return None
        self.db.delete(db_project)
        self.db.commit()
        return db_project

   # Image CRUD
    def create_image(self, image: schemas.ImageCreate, project_id: int):
        db_image = models.Image(**image.dict(), project_id=project_id)
        self.db.add(db_image)
        self.db.commit()
        self.db.refresh(db_image)
        return db_image

    def get_images(self, project_id: int):
        return self.db.query(models.Image).filter(models.Image.project_id == project_id).all()

    def get_image(self, image_id: int):
        return self.db.query(models.Image).filter(models.Image.id == image_id).first()

    def update_image(self, image_id: int, image_update: schemas.ImageUpdate):
        db_image = self.get_image(image_id)
        if not db_image:
            return None
        for key, value in image_update.dict(exclude_unset=True).items():
            setattr(db_image, key, value)
        self.db.commit()
        self.db.refresh(db_image)
        return db_image

    def delete_image(self, image_id: int):
        db_image = self.get_image(image_id)
        if not db_image:
            return None
        self.db.delete(db_image)
        self.db.commit()
        return db_image

    # Mask CRUD
    def create_mask(self, mask: schemas.MaskCreate, image_id: int):
        db_mask = models.Mask(**mask.dict(), image_id=image_id)
        self.db.add(db_mask)
        self.db.commit()
        self.db.refresh(db_mask)
        return db_mask

    def get_masks(self, image_id: int):
        return self.db.query(models.Mask).filter(models.Mask.image_id == image_id).all()

    def get_mask(self, mask_id: int):
        return self.db.query(models.Mask).filter(models.Mask.id == mask_id).first()

    def update_mask(self, mask_id: int, mask_update: schemas.MaskUpdate):
        db_mask = self.get_mask(mask_id)
        if db_mask is None:
            return None
        for key, value in mask_update.dict(exclude_unset=True).items():
            setattr(db_mask, key, value)
        self.db.commit()
        self.db.refresh(db_mask)
        return db_mask

    def delete_mask(self, mask_id: int):
        db_mask = self.get_mask(mask_id)
        if db_mask is None:
            return None
        self.db.delete(db_mask)
        self.db.commit()
        return db_mask

    def get_images_for_project(self, project_id: int):
        """
        Devuelve todas las imágenes asociadas a un proyecto específico junto con sus máscaras.
        """
        return (
            self.db.query(models.Image)
            .filter(models.Image.project_id == project_id)
            .all()
        )

    def get_masks_for_image(self, image_id: int):
        """
        Devuelve todas las máscaras asociadas a una imagen específica.

        - **image_id**: ID de la imagen.
        """
        return self.db.query(models.Mask).filter(models.Mask.image_id == image_id).all()

    def get_user_by_email(self, email: str):
        return self.db.query(models.User).filter(models.User.email == email).first()
