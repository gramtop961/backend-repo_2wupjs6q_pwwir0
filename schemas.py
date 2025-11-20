"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
Each Pydantic model represents a collection in your database.
Model name lowercased is used as the collection name.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

# Core user profile stored by JWT subject (phone/email) as user_id
class Profile(BaseModel):
    user_id: str = Field(..., description="JWT subject identifier")
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    preferred_language: Optional[str] = Field(default="en")
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    occupation: Optional[str] = None
    land_size: Optional[float] = None

class ChatMessage(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    role: str = Field(..., description="user or assistant")
    content: str
    language: Optional[str] = "en"
    state: Optional[str] = None
    district: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    confidence: Optional[float] = None
    verified: Optional[bool] = None

class Complaint(BaseModel):
    user_id: str
    text: str
    image_url: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    location: Optional[str] = None
    status: str = "filed"
    timeline: List[Dict[str, str]] = []

class Bookmark(BaseModel):
    user_id: str
    kind: str = Field(..., description="e.g., scheme")
    ref_id: str = Field(..., description="external reference id like scheme id")
    data: Dict[str, Any] = {}

class RagDoc(BaseModel):
    user_id: Optional[str] = None  # global if None
    source_type: str = Field(..., description="text|url|pdf|youtube|csv")
    title: Optional[str] = None
    url: Optional[str] = None
    language: Optional[str] = "en"
    tags: List[str] = []
    chunk: str = Field(..., description="stored content chunk")
    chunk_index: int = 0
    doc_id: Optional[str] = None  # logical doc grouping id

# Helper envelope for timestamps (not enforced here; database.py adds timestamps)
class Timestamped(BaseModel):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
