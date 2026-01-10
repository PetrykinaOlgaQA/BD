from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PostBase(BaseModel):
    content: str
    topic_id: int
    user_id: int

class PostCreate(PostBase):
    pass

class Post(PostBase):
    id: int
    created_date: datetime

    class Config:
        from_attributes = True