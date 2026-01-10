from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_date: datetime

    class Config:
        from_attributes = True  # для SQLAlchemy 2.0+
        # Если используешь SQLAlchemy 1.4 — замени на: orm_mode = True