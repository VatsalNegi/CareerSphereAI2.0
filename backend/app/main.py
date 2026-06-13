from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

from app.database.connection import engine, Base, SessionLocal, get_db
from app.models.user_model import User
from app.utils.auth_utils import get_password_hash
from app.routes import career, chatbot, mental_health, burnout, auth, admin
from app.services.model_loader import download_models_if_needed

# Download ML models from Hugging Face Hub if not present (for cloud deployment)
download_models_if_needed()

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="CareerSphere AI 2.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
# Path is relative to this file: ../../frontend
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Seed Admin User
def seed_admin():
    db = SessionLocal()
    admin_email = "vatsalnegi412@gmail.com"
    admin = db.query(User).filter(User.email == admin_email).first()
    if not admin:
        hashed_password = get_password_hash("vatsal@07")
        admin_user = User(
            name="Vatsal Negi",
            email=admin_email,
            password=hashed_password,
            role="admin",
            status="approved"
        )
        db.add(admin_user)
        db.commit()
    db.close()

seed_admin()

# Include Routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(career.router)
app.include_router(chatbot.router)
app.include_router(mental_health.router)
app.include_router(burnout.router)

@app.get("/")
def home():
    return {"message": "Welcome to CareerSphere AI 2.0 API"}