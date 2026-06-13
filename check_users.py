from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os

# Adjust path to find the database and models
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from app.database.connection import SQLALCHEMY_DATABASE_URL
from app.models.user_model import User

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

users = db.query(User).all()
for u in users:
    print(f"ID: {u.id}, Email: {u.email}, Role: {u.role}, Status: {u.status}")

db.close()
