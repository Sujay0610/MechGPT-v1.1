#!/usr/bin/env python3
"""
Startup script for MechAgent FastAPI backend
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def main():
    import uvicorn
    from main import app
    
    # Configuration from environment variables
    host = os.getenv('FASTAPI_HOST', '0.0.0.0')
    port = int(os.getenv('FASTAPI_PORT', 8000))
    reload = os.getenv('FASTAPI_RELOAD', 'true').lower() == 'true'
    
    print(f"Starting MechAgent FastAPI backend...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"API Documentation: http://{host}:{port}/docs")
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('parsed', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()