from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db, User
from auth import get_password_hash, verify_password, create_access_token, get_current_user, create_reset_token, verify_reset_token
from email_service import send_email_html

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "farmer"


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(400, "Email already registered")
    if req.role not in ("farmer", "commissioner"):
        raise HTTPException(400, "Role must be 'farmer' or 'commissioner'")
    user = User(
        email=req.email,
        password_hash=get_password_hash(req.password),
        full_name=req.full_name,
        role=req.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"token": token, "user": {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}}


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"token": token, "user": {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}}


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@router.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        # Return generic success to prevent email enumeration
        return {"message": "Если этот email зарегистрирован, на него была отправлена ссылка для сброса пароля."}
    
    token = create_reset_token(user.email)
    
    base_url = str(request.base_url).rstrip('/')
    reset_url = f"{base_url}/auth?reset_token={token}"
    
    html_content = f'''
    <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>Восстановление пароля</h2>
            <p>Вы (или кто-то другой) запросили сброс пароля для аккаунта <b>{user.email}</b>.</p>
            <p>Для создания нового пароля перейдите по ссылке ниже:</p>
            <div style="margin: 30px 0;">
                <a href="{reset_url}" style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">Сбросить пароль</a>
            </div>
            <p>Ссылка действительна в течение 1 часа. Если вы не запрашивали сброс, просто проигнорируйте это письмо.</p>
        </body>
    </html>
    '''
    send_email_html(user.email, "Восстановление пароля", html_content)
    
    return {"message": "Если этот email зарегистрирован, на него была отправлена ссылка для сброса пароля."}


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    email = verify_reset_token(req.token)
    if not email:
        raise HTTPException(400, "Токен недействителен или истек.")
        
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(400, "Пользователь не найден.")
        
    user.password_hash = get_password_hash(req.new_password)
    db.commit()
    return {"message": "Пароль успешно обновлен. Теперь вы можете войти."}


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    profile = current_user.farmer_profile
    profile_data = None
    if profile:
        profile_data = {
            "id": profile.id, "farm_name": profile.farm_name,
            "region": profile.region, "district": profile.district,
            "livestock_direction": profile.livestock_direction,
            "heads_count": profile.heads_count,
            "years_active": profile.years_active,
        }
    return {
        "id": current_user.id, "email": current_user.email,
        "full_name": current_user.full_name, "role": current_user.role,
        "profile": profile_data,
    }
