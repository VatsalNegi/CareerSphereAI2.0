from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os

# Adjust path to find the models
# Assuming run from backend/
from app.database.connection import SQLALCHEMY_DATABASE_URL
from app.models.user_model import User

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

try:
    users = db.query(User).all()
    print("--- User List ---")
    for u in users:
        print(f"ID: {u.id}, Email: {u.email}, Role: {u.role}, Status: {u.status}")
except Exception as e:
    print(f"Error: {e}")
finally:
    db.close()
