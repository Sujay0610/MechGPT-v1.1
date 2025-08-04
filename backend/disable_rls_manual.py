#!/usr/bin/env python3
"""
Manual script to disable RLS using direct SQL execution
"""

import os
import psycopg2
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

def disable_rls_manually():
    """Disable RLS by connecting directly to PostgreSQL"""
    try:
        # Parse Supabase URL to get connection details
        supabase_url = os.getenv("SUPABASE_URL")
        if not supabase_url:
            print("❌ SUPABASE_URL not found")
            return False
        
        # Extract host from Supabase URL
        parsed = urlparse(supabase_url)
        host = parsed.hostname
        
        # Construct PostgreSQL connection string
        # Note: This requires the database password which might not be available
        print(f"🔍 Attempting to connect to PostgreSQL at {host}")
        print("⚠️  Note: This requires direct database access which might not be available")
        
        # SQL commands to disable RLS
        sql_commands = [
            "ALTER TABLE users DISABLE ROW LEVEL SECURITY;",
            "ALTER TABLE agents DISABLE ROW LEVEL SECURITY;", 
            "ALTER TABLE conversations DISABLE ROW LEVEL SECURITY;",
            "ALTER TABLE messages DISABLE ROW LEVEL SECURITY;",
            "DROP POLICY IF EXISTS \"Users can view own data\" ON users;",
            "DROP POLICY IF EXISTS \"Users can update own data\" ON users;",
            "DROP POLICY IF EXISTS \"Users can create own agents\" ON agents;",
            "DROP POLICY IF EXISTS \"Users can view own agents\" ON agents;",
            "DROP POLICY IF EXISTS \"Users can update own agents\" ON agents;",
            "DROP POLICY IF EXISTS \"Users can delete own agents\" ON agents;",
            "DROP POLICY IF EXISTS \"Users can create own conversations\" ON conversations;",
            "DROP POLICY IF EXISTS \"Users can view own conversations\" ON conversations;",
            "DROP POLICY IF EXISTS \"Users can update own conversations\" ON conversations;",
            "DROP POLICY IF EXISTS \"Users can delete own conversations\" ON conversations;",
            "DROP POLICY IF EXISTS \"Users can create own messages\" ON messages;",
            "DROP POLICY IF EXISTS \"Users can view own messages\" ON messages;"
        ]
        
        print("\n📋 SQL commands to run manually in Supabase SQL Editor:")
        print("=" * 60)
        for cmd in sql_commands:
            print(cmd)
        print("=" * 60)
        
        print("\n💡 Instructions:")
        print("1. Go to your Supabase dashboard")
        print("2. Navigate to SQL Editor")
        print("3. Copy and paste the above SQL commands")
        print("4. Execute them one by one or all at once")
        print("5. Restart your backend server")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Manual RLS Disable Script\n")
    disable_rls_manually()
    print("\n🏁 Script completed.")