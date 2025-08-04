from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import os

from models.auth_schemas import (
    UserCreate, UserLogin, OTPRequest, OTPVerify, 
    PasswordReset, AuthResponse, Token, User
)
from services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()
auth_service = AuthService()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    """Get current authenticated user"""
    token = credentials.credentials
    user_data = auth_service.verify_token(token)
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_data

@router.post("/register", response_model=AuthResponse)
async def register(user_data: UserCreate):
    """Register a new user"""
    try:
        result = await auth_service.register_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        if result['success']:
            return AuthResponse(
                success=True,
                message=result['message'],
                data=result.get('data')
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['message']
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=AuthResponse)
async def login(login_data: UserLogin):
    """Login user"""
    try:
        result = await auth_service.login_user(
            email=login_data.email,
            password=login_data.password
        )
        
        if result['success']:
            return AuthResponse(
                success=True,
                message=result['message'],
                data=result.get('data')
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result['message']
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.post("/send-verification-otp", response_model=AuthResponse)
async def send_verification_otp(otp_request: OTPRequest):
    """Send verification OTP to email"""
    try:
        result = await auth_service.send_verification_otp(otp_request.email)
        
        if result['success']:
            return AuthResponse(
                success=True,
                message=result['message']
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['message']
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send OTP: {str(e)}"
        )

@router.post("/verify-email", response_model=AuthResponse)
async def verify_email(verify_data: OTPVerify):
    """Verify email with OTP"""
    try:
        result = await auth_service.verify_email(
            email=verify_data.email,
            otp_code=verify_data.otp_code
        )
        
        if result['success']:
            return AuthResponse(
                success=True,
                message=result['message']
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['message']
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email verification failed: {str(e)}"
        )

@router.post("/send-password-reset-email", response_model=AuthResponse)
async def send_password_reset_email(otp_request: OTPRequest):
    """Send password reset email"""
    try:
        result = await auth_service.send_password_reset_otp(otp_request.email)
        
        if result['success']:
            return AuthResponse(
                success=True,
                message=result['message']
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['message']
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send password reset email: {str(e)}"
        )

@router.post("/reset-password", response_model=AuthResponse)
async def reset_password(reset_data: PasswordReset, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Reset password using access token"""
    try:
        # Extract access token from the authorization header
        access_token = credentials.credentials
        
        result = await auth_service.reset_password(
            access_token=access_token,
            new_password=reset_data.new_password
        )
        
        if result['success']:
            return AuthResponse(
                success=True,
                message=result['message']
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['message']
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password reset failed: {str(e)}"
        )

@router.get("/me", response_model=AuthResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    try:
        user_info = {
            'id': current_user['id'],
            'email': current_user['email'],
            'full_name': current_user['full_name'],
            'is_verified': current_user['is_verified'],
            'created_at': current_user['created_at'],
            'updated_at': current_user['updated_at'],
            'last_login': current_user.get('last_login')
        }
        
        return AuthResponse(
            success=True,
            message="User information retrieved successfully",
            data={'user': user_info}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user info: {str(e)}"
        )

@router.post("/logout", response_model=AuthResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user (client should remove token)"""
    return AuthResponse(
        success=True,
        message="Logged out successfully"
    )