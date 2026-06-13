from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from app.database.connection import get_db
from app.models.user_model import User
from app.schemas.auth_schema import UserOut
from app.utils.auth_utils import get_admin_user

router = APIRouter(prefix="/admin", tags=["admin"])

class StatusUpdate(BaseModel):
    email: str
    status: str

@router.get("/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db), admin: User = Depends(get_admin_user)):
    return db.query(User).all()

@router.post("/approve")
def approve_user_v2(data: StatusUpdate, db: Session = Depends(get_db), admin: User = Depends(get_admin_user)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.status = data.status
    db.commit()
    return {"message": f"User {user.email} updated to {data.status}"}

@router.post("/approve/{user_id}")
def approve_user(user_id: int, db: Session = Depends(get_db), admin: User = Depends(get_admin_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.status = "approved"
    db.commit()
    return {"message": f"User {user.email} approved"}

@router.post("/reject/{user_id}")
def reject_user(user_id: int, db: Session = Depends(get_db), admin: User = Depends(get_admin_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.status = "rejected"
    db.commit()
    return {"message": f"User {user.email} rejected"}
