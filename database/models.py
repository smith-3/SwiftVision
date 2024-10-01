from sqlalchemy import Column, Integer, String, ForeignKey, Text, Float, TIMESTAMP, func
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
    base_image = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

    project = relationship("Project", back_populates="images")
    annotations = relationship("Annotation", back_populates="image")

class Annotation(Base):
    __tablename__ = 'annotations'

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    segmentation = Column(Text, nullable=False)
    bbox = Column(String(255), nullable=True)
    area = Column(Integer, nullable=True)
    predicted_iou = Column(Float, nullable=True)
    stability_score = Column(Float, nullable=True)
    crop_box = Column(String(255), nullable=True)
    point_coords = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    image = relationship("Image", back_populates="annotations")
