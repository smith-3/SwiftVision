from sqlalchemy import Column, Integer, LargeBinary, String, ForeignKey, Text, Float, TIMESTAMP, func
from sqlalchemy.orm import relationship
from .database import Base
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

    projects = relationship("Project", back_populates="user")

class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(TIMESTAMP, server_default=func.now())

    user = relationship("User", back_populates="projects")
    images = relationship("Image", back_populates="project")

class Image(Base):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    base_image = Column(LargeBinary, nullable=False)  # Contenido binario de la imagen
    created_at = Column(TIMESTAMP, server_default=func.now())

    project = relationship("Project", back_populates="images")
    masks = relationship("Mask", back_populates="image")

class Mask(Base):
    __tablename__ = 'masks'

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    counts = Column(Text, nullable=False)
    size = Column(String(255), nullable=True)
    bbox = Column(String(255), nullable=True)
    point_coords = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    image = relationship("Image", back_populates="masks")
