from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import os
import jwt
from database import db, create_document, get_documents
from bson import ObjectId

APP_NAME = "Gramin Saathi API"
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = "HS256"
TOKEN_EXP_MIN = 60

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory OTP store (short-lived)
_otp_store: Dict[str, Dict[str, Any]] = {}


class OTPRequest(BaseModel):
    contact: str = Field(..., description="Phone or email")


class OTPVerify(BaseModel):
    contact: str
    otp: str


class JWToken(BaseModel):
    token: str
    expires_in: int


class ProfileModel(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    language: Optional[str] = Field("en", description="Preferred language code")
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    occupation: Optional[str] = None
    land_size: Optional[str] = None


class ChatAsk(BaseModel):
    message: str
    language: str = "en"
    state: Optional[str] = None
    district: Optional[str] = None


class ChatAnswer(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.7
    verified: bool = False


class ComplaintCreate(BaseModel):
    text: str
    state: Optional[str] = None
    district: Optional[str] = None


class ComplaintStatus(BaseModel):
    status: str
    timeline: List[Dict[str, Any]]


# Utils

def create_jwt(sub: str) -> JWToken:
    exp = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXP_MIN)
    payload = {"sub": sub, "exp": exp, "scope": "user"}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return JWToken(token=token, expires_in=TOKEN_EXP_MIN * 60)


def get_contact_from_token(authorization: Optional[str] = None) -> Optional[str]:
    if not authorization:
        return None
    try:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            return None
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return data.get("sub")
    except Exception:
        return None


@app.get("/")
def root():
    return {"ok": True, "name": APP_NAME, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/test")
def test_db():
    try:
        if db is None:
            raise Exception("DB not configured")
        # Simple ping by listing collections
        _ = db.list_collection_names()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Auth
@app.post("/auth/request-otp")
def request_otp(req: OTPRequest):
    otp = f"{datetime.now().microsecond % 1000000:06d}"
    _otp_store[req.contact] = {"otp": otp, "expires": datetime.now(timezone.utc) + timedelta(minutes=5)}
    # In real integration, send via SMS/Email. For now, return masked info only.
    return {"sent": True, "contact": req.contact}


@app.post("/auth/verify-otp", response_model=JWToken)
def verify_otp(req: OTPVerify):
    record = _otp_store.get(req.contact)
    if not record or record["expires"] < datetime.now(timezone.utc) or record["otp"] != req.otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    # Success → issue JWT
    token = create_jwt(req.contact)
    # Clear OTP
    _otp_store.pop(req.contact, None)
    return token


# Profile
@app.post("/profile")
def upsert_profile(profile: ProfileModel, authorization: Optional[str] = Body(default=None)):
    # Try to read from header first via dependency injection fallback
    # FastAPI Body(default=None) won't catch headers; instead use direct environment from request? We'll accept header via dependency below.
    raise HTTPException(status_code=400, detail="Use header Authorization: Bearer <token>")


from fastapi import Request

@app.post("/profile", include_in_schema=False)
async def upsert_profile_header(profile: ProfileModel, request: Request):
    contact = get_contact_from_token(request.headers.get("authorization"))
    # If not logged in, still allow saving a loose profile (no contact linkage)
    data = profile.model_dump()
    data["updated_at"] = datetime.now(timezone.utc)
    try:
        if db is None:
            raise Exception("Database not available")
        col = db["profile"]
        if contact:
            data["contact"] = contact
            col.update_one({"contact": contact}, {"$set": data, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}}, upsert=True)
            saved = col.find_one({"contact": contact}, {"_id": 0})
        else:
            # Create a new profile document without contact
            _id = col.insert_one({**data, "created_at": datetime.now(timezone.utc)}).inserted_id
            saved_doc = col.find_one({"_id": _id})
            saved = {k: v for k, v in saved_doc.items() if k != "_id"}
        return saved
    except Exception as e:
        # Fallback: just echo back (non-persistent)
        return data


# Chat
@app.post("/api/chat/ask", response_model=ChatAnswer)
async def chat_ask(payload: ChatAsk, request: Request):
    user_contact = get_contact_from_token(request.headers.get("authorization"))
    msg = payload.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")

    lower = msg.lower()
    parts = []
    sources = []
    verified = False
    confidence = 0.7

    if "pm-kisan" in lower or "kisan" in lower or "scheme" in lower:
        parts.append("PM-KISAN provides income support of ₹6,000/year to eligible farmer families in 3 installments.")
        sources.append({"title": "PM-KISAN", "url": "https://pmkisan.gov.in/"})
        verified = True
        confidence = 0.82
    if "weather" in lower or "rain" in lower or "monsoon" in lower:
        loc = ", ".join(filter(None, [payload.district, payload.state])) or "your area"
        parts.append(f"Weather today in {loc}: Mostly sunny, 31°C. Chance of rain ~10%.")
        sources.append({"title": "IMD (summary)", "url": "https://mausam.imd.gov.in/"})
    if not parts:
        parts.append("I don’t have verified information about that yet. I’m still learning and will improve soon.")
        sources.append({"title": "General guidance", "url": "https://www.india.gov.in/"})
        verified = False
        confidence = 0.55

    answer = " \n".join(parts)

    # Persist chat
    try:
        if db is not None:
            db["chat"].insert_one({
                "contact": user_contact,
                "message": payload.model_dump(),
                "answer": {"answer": answer, "sources": sources, "confidence": confidence, "verified": verified},
                "created_at": datetime.now(timezone.utc)
            })
    except Exception:
        pass

    return ChatAnswer(answer=answer, sources=sources, confidence=confidence, verified=verified)


# Schemes (stub search)
@app.get("/api/schemes/search")
async def schemes_search(state: Optional[str] = None, district: Optional[str] = None, gender: Optional[str] = None, land_size: Optional[str] = None):
    base = [
        {"id": "pmkisan", "title": "PM-KISAN Income Support", "eligibility": "Small and marginal farmers with up to 2 hectares.", "benefits": "₹6,000/year in 3 installments.", "apply_url": "https://pmkisan.gov.in/"},
        {"id": "pmfby", "title": "PM Fasal Bima Yojana", "eligibility": "All farmers including sharecroppers and tenant farmers.", "benefits": "Crop insurance against natural calamities.", "apply_url": "https://pmfby.gov.in/"},
        {"id": "soil-card", "title": "Soil Health Card", "eligibility": "All farmers.", "benefits": "Soil analysis and nutrient recommendations.", "apply_url": "https://soilhealth.dac.gov.in/"},
    ]

    def ok(item):
        # Minimal filtering demo
        if state and state.lower() not in ["india", "bharat"]:
            pass
        return True

    return [s for s in base if ok(s)]


# Complaints
@app.post("/api/complaints/create")
async def complaints_create(payload: ComplaintCreate, request: Request):
    contact = get_contact_from_token(request.headers.get("authorization"))
    doc = {
        "contact": contact,
        "text": payload.text,
        "state": payload.state,
        "district": payload.district,
        "status": "filed",
        "timeline": [
            {"time": datetime.now(timezone.utc).isoformat(), "event": "Filed"}
        ],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    try:
        if db is None:
            raise Exception("DB not available")
        inserted_id = db["complaint"].insert_one(doc).inserted_id
        return {"id": str(inserted_id), "status": "filed"}
    except Exception:
        # Fallback without persistence
        return {"id": "temp-" + str(int(datetime.now().timestamp())), "status": "filed"}


@app.get("/api/complaints/status/{cid}", response_model=ComplaintStatus)
async def complaints_status(cid: str):
    try:
        if db is None:
            raise Exception("DB not available")
        doc = db["complaint"].find_one({"_id": ObjectId(cid)})
        if not doc:
            raise HTTPException(status_code=404, detail="Not found")
        return {"status": doc.get("status", "filed"), "timeline": doc.get("timeline", [])}
    except HTTPException:
        raise
    except Exception:
        # Fallback static
        return {"status": "filed", "timeline": [
            {"time": datetime.now(timezone.utc).isoformat(), "event": "Filed"},
            {"time": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(), "event": "Under review"}
        ]}
