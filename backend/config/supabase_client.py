import os
from supabase import create_client, Client
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SupabaseClient:
    _instance: Optional['SupabaseClient'] = None
    _client: Optional[Client] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SupabaseClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client with environment variables"""
        supabase_url = os.getenv("SUPABASE_URL")
        # Try service role key first (for backend operations), fallback to anon key
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        anon_key = os.getenv("SUPABASE_ANON_KEY")
        
        # Use service role key for backend operations to bypass RLS
        supabase_key = service_role_key if service_role_key else anon_key
        key_type = "service role" if service_role_key else "anon"
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and either SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY must be set in environment variables"
            )
        
        self._client = create_client(supabase_url, supabase_key)
        print(f"Supabase client initialized successfully with {key_type} key")
    
    @property
    def client(self) -> Client:
        """Get the Supabase client instance"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def get_client(self) -> Client:
        """Get the Supabase client instance (alternative method)"""
        return self.client

# Global instance
supabase_client = SupabaseClient()

# Convenience function
def get_supabase_client() -> Client:
    """Get the global Supabase client instance"""
    return supabase_client.get_client()