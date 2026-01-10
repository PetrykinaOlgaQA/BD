from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CommentBase(BaseModel):
    content: str
    post_id: int
    user_id: int

class CommentCreate(CommentBase):
    pass

class Comment(CommentBase):
    id: int
    created_date: datetime

    class Config:
        from_attributes = True