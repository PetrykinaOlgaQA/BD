from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TopicBase(BaseModel):
    title: str
    description: Optional[str] = None
    user_id: int

class TopicCreate(TopicBase):
    pass

class Topic(TopicBase):
    id: int
    created_date: datetime

    class Config:
        from_attributes = True