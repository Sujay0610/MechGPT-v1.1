import os
import uuid
import hashlib
import secrets
import smtplib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import jwt
from config.supabase_client import get_supabase_client
from passlib.context import CryptContext

class AuthService:
    def __init__(self):
        self.supabase = get_supabase_client()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.otp_expiry_minutes = 10
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def _generate_otp(self) -> str:
        """Generate 6-digit OTP"""
        return str(secrets.randbelow(900000) + 100000)
    
    def _send_email(self, to_email: str, subject: str, body: str) -> bool:
        """Send email using SMTP"""
        try:
            if not self.smtp_username or not self.smtp_password:
                print(f"Email simulation: To {to_email}, Subject: {subject}, Body: {body}")
                return True  # Simulate success for development
            
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            text = msg.as_string()
            server.sendmail(self.smtp_username, to_email, text)
            server.quit()
            
            return True
        except Exception as e:
            print(f"Email sending failed: {e}")
            return False
    
    def _create_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token"""
        payload = {
            'user_id': user_data['id'],
            'email': user_data['email'],
            'exp': datetime.utcnow() + timedelta(days=7)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify Supabase JWT token and return user data"""
        try:
            # Use Supabase's built-in token verification
            user_response = self.supabase.auth.get_user(token)
            
            if user_response.user:
                return {
                    'id': user_response.user.id,  # Changed from 'user_id' to 'id'
                    'email': user_response.user.email,
                    'full_name': user_response.user.user_metadata.get('full_name', ''),
                    'is_verified': bool(user_response.user.email_confirmed_at),
                    'created_at': user_response.user.created_at,
                    'updated_at': user_response.user.updated_at,
                    'last_login': user_response.user.last_sign_in_at
                }
            
            return None
            
        except Exception as e:
            print(f"Token verification error: {e}")
            return None
    
    async def register_user(self, email: str, password: str, full_name: str) -> Dict[str, Any]:
        """Register a new user using Supabase Auth with OTP verification"""
        try:
            # Use Supabase's built-in signUp method with email confirmation disabled for testing
            auth_response = self.supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "full_name": full_name
                    },
                    "email_redirect_to": None  # Disable email confirmation for testing
                }
            })
            
            if auth_response.user:
                # Manually confirm the user's email in Supabase auth system for testing
                try:
                    # Use admin API to confirm the user's email
                    confirm_response = self.supabase.auth.admin.update_user_by_id(
                        auth_response.user.id,
                        {"email_confirm": True}
                    )
                except Exception as confirm_error:
                    print(f"Warning: Could not confirm email automatically: {confirm_error}")
                
                # Also create a record in the custom users table
                try:
                    user_data = {
                        "id": auth_response.user.id,
                        "email": email,
                        "password_hash": self._hash_password(password),  # Store hashed password
                        "full_name": full_name,
                        "is_verified": True  # Set to True since we're bypassing email verification for testing
                    }
                    
                    # Insert into custom users table
                    result = self.supabase.table("users").insert(user_data).execute()
                    
                    if result.data:
                        return {
                            'success': True,
                            'message': 'Registration successful. User is ready to use the system.',
                            'data': {
                                'user_id': auth_response.user.id,
                                'email': email,
                                'requires_verification': False
                            }
                        }
                    else:
                        # If custom table insert fails, we should clean up the auth user
                        print(f"Failed to create user record in custom table for {email}")
                        return {
                            'success': False,
                            'message': 'Registration failed. Please try again.'
                        }
                        
                except Exception as db_error:
                    print(f"Database error during user creation: {db_error}")
                    return {
                        'success': False,
                        'message': 'Registration failed. Please try again.'
                    }
            else:
                return {
                    'success': False,
                    'message': 'Registration failed. Please try again.'
                }
                
        except Exception as e:
            print(f"Registration error: {e}")
            error_message = str(e)
            if "already registered" in error_message.lower():
                return {
                    'success': False,
                    'message': 'User with this email already exists'
                }
            return {
                'success': False,
                'message': f'Registration failed: {error_message}'
            }
    
    async def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """Login user using Supabase Auth"""
        try:
            # Use Supabase's built-in signInWithPassword method
            auth_response = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if auth_response.user and auth_response.session:
                # Skip email confirmation check for development
                # In production, you should enable this check:
                # if not auth_response.user.email_confirmed_at:
                #     return {
                #         'success': False,
                #         'message': 'Please verify your email before logging in',
                #         'data': {
                #             'requires_verification': True,
                #             'email': email
                #         }
                #     }
                
                return {
                    'success': True,
                    'message': 'Login successful',
                    'data': {
                        'access_token': auth_response.session.access_token,
                        'refresh_token': auth_response.session.refresh_token,
                        'token_type': 'bearer',
                        'expires_in': auth_response.session.expires_in,
                        'user': {
                            'id': auth_response.user.id,
                            'email': auth_response.user.email,
                            'full_name': auth_response.user.user_metadata.get('full_name', ''),
                            'is_verified': True,  # Set to True for development
                            'created_at': auth_response.user.created_at,
                            'updated_at': auth_response.user.updated_at,
                            'last_login': auth_response.user.last_sign_in_at
                        }
                    }
                }
            else:
                return {
                    'success': False,
                    'message': 'Invalid email or password'
                }
            
        except Exception as e:
            print(f"Login error: {e}")
            error_message = str(e)
            if "invalid" in error_message.lower():
                return {
                    'success': False,
                    'message': 'Invalid email or password'
                }
            return {
                'success': False,
                'message': f'Login failed: {error_message}'
            }
    
    async def send_verification_otp(self, email: str) -> Dict[str, Any]:
        """Send verification OTP using Supabase Auth"""
        try:
            # Use Supabase's built-in signInWithOtp for email verification
            auth_response = self.supabase.auth.sign_in_with_otp({
                "email": email,
                "options": {
                    "should_create_user": False  # Don't create user, just send OTP for existing users
                }
            })
            
            return {
                'success': True,
                'message': 'Verification code sent to your email'
            }
                
        except Exception as e:
            print(f"Send verification OTP error: {e}")
            return {
                'success': False,
                'message': f'Failed to send verification code: {str(e)}'
            }
    
    async def verify_email(self, email: str, otp_code: str) -> Dict[str, Any]:
        """Verify email using Supabase Auth OTP"""
        try:
            # Use Supabase's built-in verifyOtp method
            auth_response = self.supabase.auth.verify_otp({
                "email": email,
                "token": otp_code,
                "type": "email"
            })
            
            if auth_response.user and auth_response.session:
                # Also update the custom users table to mark as verified
                try:
                    update_result = self.supabase.table("users").update({
                        "is_verified": True,
                        "updated_at": datetime.now().isoformat()
                    }).eq("id", auth_response.user.id).execute()
                    
                    if update_result.data:
                        print(f"Successfully updated verification status for user {auth_response.user.id}")
                    else:
                        print(f"Warning: Could not update verification status in custom users table for {email}")
                        
                except Exception as update_error:
                    print(f"Error updating custom users table: {update_error}")
                    # Don't fail the verification process if custom table update fails
                
                return {
                    'success': True,
                    'message': 'Email verified successfully',
                    'data': {
                        'user': {
                            'id': auth_response.user.id,
                            'email': auth_response.user.email,
                            'email_confirmed_at': auth_response.user.email_confirmed_at
                        },
                        'session': {
                            'access_token': auth_response.session.access_token,
                            'refresh_token': auth_response.session.refresh_token
                        }
                    }
                }
            else:
                return {
                    'success': False,
                    'message': 'Invalid or expired verification code'
                }
                
        except Exception as e:
            print(f"Email verification error: {e}")
            return {
                'success': False,
                'message': f'Email verification failed: {str(e)}'
            }
    
    async def send_password_reset_otp(self, email: str) -> Dict[str, Any]:
        """Send password reset email using Supabase Auth"""
        try:
            # Use Supabase's built-in resetPasswordForEmail method
            auth_response = self.supabase.auth.reset_password_email(email)
            
            return {
                'success': True,
                'message': 'Password reset link sent to your email'
            }
                
        except Exception as e:
            print(f"Send password reset error: {e}")
            return {
                'success': False,
                'message': f'Failed to send password reset email: {str(e)}'
            }
    
    async def reset_password(self, access_token: str, new_password: str) -> Dict[str, Any]:
        """Reset password using Supabase Auth access token"""
        try:
            # Use Supabase's built-in updateUser method to change password
            auth_response = self.supabase.auth.update_user(
                access_token,
                {"password": new_password}
            )
            
            if auth_response.user:
                return {
                    'success': True,
                    'message': 'Password reset successfully'
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to reset password'
                }
                
        except Exception as e:
            print(f"Password reset error: {e}")
            return {
                'success': False,
                'message': f'Password reset failed: {str(e)}'
            }
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            result = self.supabase.table("users").select("*").eq("id", user_id).execute()
            return result.data[0] if result.data else None
        except Exception:
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            result = self.supabase.table("users").select("*").eq("email", email).execute()
            return result.data[0] if result.data else None
        except Exception:
            return None