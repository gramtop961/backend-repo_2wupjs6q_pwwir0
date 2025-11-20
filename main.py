import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import jwt

# Simple in-memory OTP store (for demo). In production, use DB/Redis.
otp_store: Dict[str, Dict[str, Any]] = {}
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = "HS256"

app = FastAPI(title="Gramin Saathi API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuthRequest(BaseModel):
    identifier: str  # email or phone
    channel: str = "sms"  # sms or email

class VerifyRequest(BaseModel):
    identifier: str
    otp: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600

class Profile(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    preferred_language: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    occupation: Optional[str] = None
    land_size: Optional[float] = None

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "en"
    state: Optional[str] = None
    district: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.6
    verified: bool = False

class Scheme(BaseModel):
    id: str
    title: str
    state: Optional[str] = None
    district: Optional[str] = None
    eligibility: str
    benefits: str
    apply_url: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    """Minimal health for now"""
    return {"backend": "✅ Running"}

# ---------------------- AUTH ----------------------
@app.post("/auth/request-otp", response_model=dict)
def request_otp(payload: AuthRequest):
    import random
    code = f"{random.randint(100000, 999999)}"
    # store with 5-min expiry
    otp_store[payload.identifier] = {"code": code, "exp": datetime.utcnow() + timedelta(minutes=5)}
    # In production: send via SMS/email provider
    return {"sent": True, "channel": payload.channel, "debug_code": code}

@app.post("/auth/verify-otp", response_model=TokenResponse)
def verify_otp(payload: VerifyRequest):
    record = otp_store.get(payload.identifier)
    if not record or record["code"] != payload.otp or record["exp"] < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired OTP")
    # issue JWT
    claims = {"sub": payload.identifier, "iat": datetime.utcnow(), "exp": datetime.utcnow() + timedelta(hours=1)}
    token = jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALG)
    return TokenResponse(access_token=token)

# ---------------------- CHAT ----------------------
@app.post("/api/chat/ask", response_model=ChatResponse)
def chat_ask(req: ChatRequest):
    # Placeholder RAG: return verified-only when known keywords, else fallback
    text = req.message.strip().lower()
    sources = []
    verified = False
    answer = "I don’t have verified information for that yet."
    conf = 0.4

    if any(k in text for k in ["pm-kisan", "kisan", "pmkisan"]):
        answer = "PM-KISAN provides income support of Rs. 6,000 per year to eligible farmer families, payable in three equal installments."
        sources = [{"title": "PM-KISAN Official", "url": "https://pmkisan.gov.in/"}]
        verified = True
        conf = 0.82
    elif any(k in text for k in ["weather", "rain", "rainfall"]):
        answer = "Today looks clear with a low chance of rain in most districts. For sowing decisions, check the 5-day forecast."
        sources = [{"title": "IMD Bulletin", "url": "https://mausam.imd.gov.in/"}]
        verified = True
        conf = 0.68

    return ChatResponse(answer=answer, sources=sources, confidence=conf, verified=verified)

# ---------------------- SCHEMES (stub) ----------------------
@app.get("/api/schemes/search", response_model=List[Scheme])
def search_schemes(state: Optional[str] = None, district: Optional[str] = None, gender: Optional[str] = None, land_size: Optional[float] = None):
    demo = [
        Scheme(
            id="pm-kisan",
            title="PM-KISAN Income Support",
            state=None,
            district=None,
            eligibility="Small and marginal farmer families owning cultivable land.",
            benefits="Rs. 6,000 per year in three installments.",
            apply_url="https://pmkisan.gov.in/"
        )
    ]
    return demo

# ---------------------- COMPLAINTS (stub) ----------------------
class ComplaintCreate(BaseModel):
    text: str
    image_url: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    location: Optional[str] = None

class ComplaintStatus(BaseModel):
    id: str
    status: str
    timeline: List[Dict[str, str]]

@app.post("/api/complaints/create", response_model=ComplaintStatus)
def create_complaint(c: ComplaintCreate):
    return ComplaintStatus(
        id=f"CMP-{int(datetime.utcnow().timestamp())}",
        status="filed",
        timeline=[{"time": datetime.utcnow().isoformat(), "event": "filed"}]
    )

@app.get("/api/complaints/status/{cid}", response_model=ComplaintStatus)
def complaint_status(cid: str):
    return ComplaintStatus(
        id=cid,
        status="under_process",
        timeline=[
            {"time": (datetime.utcnow() - timedelta(days=1)).isoformat(), "event": "filed"},
            {"time": datetime.utcnow().isoformat(), "event": "under process"}
        ]
    )

# ---------------------- PROFILE ----------------------
@app.post("/profile", response_model=Profile)
def upsert_profile(profile: Profile):
    # For now, echo back. Later connect to MongoDB using provided helpers.
    return profile

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
