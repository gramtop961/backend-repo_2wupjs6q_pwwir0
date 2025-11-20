import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import jwt
from database import db, create_document
from schemas import Profile as ProfileSchema, ChatMessage, Complaint as ComplaintSchema, Bookmark, RagDoc
from bson.objectid import ObjectId

# ---------------------- AUTH/JWT ----------------------
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

# Request/response models for endpoints
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
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.6
    verified: bool = False
    session_id: Optional[str] = None

class Scheme(BaseModel):
    id: str
    title: str
    state: Optional[str] = None
    district: Optional[str] = None
    eligibility: str
    benefits: str
    apply_url: Optional[str] = None

class BookmarkIn(BaseModel):
    kind: str
    ref_id: str
    data: Dict[str, Any] = {}

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

# ---------------------- Helpers ----------------------
def get_current_user(authorization: Optional[str] = None) -> Optional[str]:
    if not authorization:
        return None
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            return None
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return claims.get("sub")
    except Exception:
        return None

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    ok = db is not None
    return {"backend": "✅ Running", "db": ok}

# ---------------------- AUTH ----------------------
@app.post("/auth/request-otp", response_model=dict)
def request_otp(payload: AuthRequest):
    import random
    code = f"{random.randint(100000, 999999)}"
    otp_store[payload.identifier] = {"code": code, "exp": datetime.utcnow() + timedelta(minutes=5)}
    return {"sent": True, "channel": payload.channel, "debug_code": code}

@app.post("/auth/verify-otp", response_model=TokenResponse)
def verify_otp(payload: VerifyRequest):
    record = otp_store.get(payload.identifier)
    if not record or record["code"] != payload.otp or record["exp"] < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired OTP")
    claims = {"sub": payload.identifier, "iat": datetime.utcnow(), "exp": datetime.utcnow() + timedelta(hours=1)}
    token = jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALG)
    return TokenResponse(access_token=token)

# ---------------------- PROFILE (Mongo) ----------------------
@app.post("/profile", response_model=Profile)
def upsert_profile(profile: Profile, authorization: Optional[str] = None):
    user_id = get_current_user(authorization)
    if not user_id:
        return profile
    doc = ProfileSchema(user_id=user_id, **profile.model_dump())
    db["profile"].update_one({"user_id": user_id}, {"$set": doc.model_dump(), "$setOnInsert": {"created_at": datetime.utcnow()}}, upsert=True)
    saved = db["profile"].find_one({"user_id": user_id}, {"_id": 0, "user_id": 0})
    return Profile(**(saved or profile.model_dump()))

# ---------------------- CHAT (with memory) ----------------------
@app.post("/api/chat/ask", response_model=ChatResponse)
def chat_ask(req: ChatRequest, authorization: Optional[str] = None):
    user_id = get_current_user(authorization)
    text = req.message.strip()
    session_id = req.session_id or f"sess-{int(datetime.utcnow().timestamp())}"
    if user_id:
        umsg = ChatMessage(user_id=user_id, session_id=session_id, role="user", content=text, language=req.language, state=req.state, district=req.district)
        create_document("chatmessage", umsg)
    lower = text.lower()
    sources: List[Dict[str, Any]] = []
    verified = False
    answer = "I don’t have verified information for that yet."
    conf = 0.4
    if any(k in lower for k in ["pm-kisan", "kisan", "pmkisan"]):
        answer = "PM-KISAN provides income support of Rs. 6,000 per year to eligible farmer families, payable in three equal installments."
        sources = [{"title": "PM-KISAN Official", "url": "https://pmkisan.gov.in/"}]
        verified = True
        conf = 0.82
    elif any(k in lower for k in ["weather", "rain", "rainfall"]):
        answer = "Today looks clear with a low chance of rain in most districts. For sowing decisions, check the 5-day forecast."
        sources = [{"title": "IMD Bulletin", "url": "https://mausam.imd.gov.in/"}]
        verified = True
        conf = 0.68
    if user_id:
        amsg = ChatMessage(user_id=user_id, session_id=session_id, role="assistant", content=answer, language=req.language, state=req.state, district=req.district, sources=sources, confidence=conf, verified=verified)
        create_document("chatmessage", amsg)
    return ChatResponse(answer=answer, sources=sources, confidence=conf, verified=verified, session_id=session_id)

@app.post("/api/chat/pin", response_model=dict)
def pin_answer(ref_id: str = Form(...), authorization: Optional[str] = None):
    user_id = get_current_user(authorization)
    if not user_id:
        raise HTTPException(401, "Unauthorized")
    create_document("bookmark", Bookmark(user_id=user_id, kind="answer", ref_id=ref_id, data={}).model_dump())
    return {"ok": True}

@app.get("/api/chat/history", response_model=List[Dict[str, Any]])
def chat_history(session_id: Optional[str] = None, authorization: Optional[str] = None):
    user_id = get_current_user(authorization)
    if not user_id:
        return []
    filt = {"user_id": user_id}
    if session_id:
        filt["session_id"] = session_id
    msgs = list(db["chatmessage"].find(filt, {"_id": 0}).sort("created_at", 1))
    return msgs

# ---------------------- SCHEMES + BOOKMARKS ----------------------
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

@app.post("/api/bookmarks", response_model=dict)
def add_bookmark(payload: BookmarkIn, authorization: Optional[str] = None):
    user_id = get_current_user(authorization)
    if not user_id:
        raise HTTPException(401, "Unauthorized")
    create_document("bookmark", Bookmark(user_id=user_id, kind=payload.kind, ref_id=payload.ref_id, data=payload.data))
    return {"ok": True}

@app.get("/api/bookmarks", response_model=List[Dict[str, Any]])
def list_bookmarks(kind: Optional[str] = None, authorization: Optional[str] = None):
    user_id = get_current_user(authorization)
    if not user_id:
        return []
    filt: Dict[str, Any] = {"user_id": user_id}
    if kind:
        filt["kind"] = kind
    docs = list(db["bookmark"].find(filt, {"_id": 0}))
    return docs

@app.delete("/api/bookmarks/{ref_id}", response_model=dict)
def delete_bookmark(ref_id: str, kind: str = "scheme", authorization: Optional[str] = None):
    user_id = get_current_user(authorization)
    if not user_id:
        raise HTTPException(401, "Unauthorized")
    db["bookmark"].delete_one({"user_id": user_id, "ref_id": ref_id, "kind": kind})
    return {"ok": True}

# ---------------------- COMPLAINTS (Mongo) ----------------------
@app.post("/api/complaints/create", response_model=ComplaintStatus)
def create_complaint(c: ComplaintCreate, authorization: Optional[str] = None):
    user_id = get_current_user(authorization)
    comp = ComplaintSchema(user_id=user_id or "anon", text=c.text, image_url=c.image_url, state=c.state, district=c.district, location=c.location, status="filed", timeline=[{"time": datetime.utcnow().isoformat(), "event": "filed"}])
    cid = create_document("complaint", comp)
    return ComplaintStatus(id=cid, status=comp.status, timeline=comp.timeline)

@app.get("/api/complaints/status/{cid}", response_model=ComplaintStatus)
def complaint_status(cid: str, authorization: Optional[str] = None):
    try:
        obj_id = ObjectId(cid)
    except Exception:
        raise HTTPException(400, "Invalid complaint id")
    doc = db["complaint"].find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(404, "Not found")
    return ComplaintStatus(id=str(doc.get("_id")), status=doc.get("status", "under_process"), timeline=doc.get("timeline", []))

# ---------------------- RAG: ingest + query (stub + DB) ----------------------
class IngestRequest(BaseModel):
    source_type: str
    title: Optional[str] = None
    url: Optional[str] = None
    language: Optional[str] = "en"
    text: Optional[str] = None
    tags: List[str] = []

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/rag/ingest", response_model=dict)
def rag_ingest(req: IngestRequest, authorization: Optional[str] = None):
    user_id = get_current_user(authorization)
    if not req.text and not req.url:
        raise HTTPException(400, "Provide text or url")
    chunk = (req.text or req.url)[:2000]
    doc = RagDoc(user_id=user_id, source_type=("url" if req.url else "text"), title=req.title, url=req.url, language=req.language, tags=req.tags, chunk=chunk, chunk_index=0)
    create_document("ragdoc", doc)
    return {"ok": True}

@app.post("/rag/query", response_model=ChatResponse)
def rag_query(req: QueryRequest, authorization: Optional[str] = None):
    q = req.question.lower()
    all_docs = list(db["ragdoc"].find({}, {"_id": 0}))
    scored = []
    for d in all_docs:
        chunk = d.get("chunk", "")
        score = sum(1 for w in q.split() if w in chunk.lower())
        if score:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return ChatResponse(answer="No relevant documents found yet. Try ingesting sources first.", sources=[], confidence=0.3, verified=False)
    top = [s[1] for s in scored[: req.top_k]]
    answer = "Based on your knowledge base: " + " ".join([t.get("chunk", "")[:140] for t in top])
    sources = [{"title": t.get("title") or t.get("url") or "ingested", "url": t.get("url") or ""} for t in top]
    return ChatResponse(answer=answer, sources=sources, confidence=0.6, verified=False)

# ---------------------- Voice STT/TTS (stubs) ----------------------
@app.post("/voice/stt", response_model=dict)
def voice_stt(file: UploadFile = File(...)):
    return {"transcript": f"Heard audio: {file.filename}"}

@app.post("/voice/tts", response_model=dict)
def voice_tts(text: str = Form(...), language: str = Form("en")):
    return {"audio_url": "", "note": "TTS stub. Use Web Speech API on frontend for now."}

# ---------------------- Farmer Tools (stubs) ----------------------
@app.get("/tools/market-prices", response_model=List[Dict[str, Any]])
def market_prices(state: Optional[str] = None, district: Optional[str] = None, commodity: Optional[str] = None):
    items = [
        {"commodity": commodity or "Tomato", "unit": "kg", "min": 8, "max": 18, "modal": 12, "market": (district or "Local Mandi"), "state": state or "KA"},
        {"commodity": commodity or "Onion", "unit": "kg", "min": 12, "max": 28, "modal": 20, "market": (district or "Local Mandi"), "state": state or "KA"}
    ]
    return items

@app.get("/tools/weather", response_model=Dict[str, Any])
def weather(state: Optional[str] = None, district: Optional[str] = None):
    return {"state": state, "district": district, "today": {"summary": "Clear", "rain_chance": 0.1, "temp_c": 30}, "next_5_days": [{"day": i+1, "rain_chance": 0.2 + i*0.05} for i in range(5)]}

class FertilizerInput(BaseModel):
    crop: str
    area_acre: float

class IrrigationInput(BaseModel):
    crop: str
    soil_type: str
    area_acre: float

@app.post("/tools/calc/fertilizer", response_model=Dict[str, Any])
def calc_fertilizer(inp: FertilizerInput):
    # Simple rule-of-thumb NPK per acre
    base = {"N": 50, "P": 25, "K": 25}
    factor = max(0.5, min(2.0, inp.area_acre))
    return {"crop": inp.crop, "area_acre": inp.area_acre, "recommendation_kg": {k: v*factor for k, v in base.items()}, "note": "Advisory only. Consult local agri officer."}

@app.post("/tools/calc/irrigation", response_model=Dict[str, Any])
def calc_irrigation(inp: IrrigationInput):
    soil_factor = {"sandy": 1.2, "loam": 1.0, "clay": 0.8}.get(inp.soil_type.lower(), 1.0)
    water_mm = 40 * soil_factor
    liters = water_mm * 10_000 * inp.area_acre / 2.471  # mm to liters over acres
    return {"crop": inp.crop, "soil_type": inp.soil_type, "area_acre": inp.area_acre, "water_liters": round(liters, 0), "schedule_days": 5}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
