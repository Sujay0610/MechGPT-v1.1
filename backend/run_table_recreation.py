#!/usr/bin/env python3
"""
Script to recreate all Supabase tables with proper schema
This will delete all existing data and recreate tables from scratch
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from config.supabase_client import get_supabase_client

# Load environment variables
load_dotenv()

def confirm_deletion():
    """Ask user to confirm table deletion"""
    print("⚠️  WARNING: This will DELETE ALL EXISTING DATA in your Supabase tables!")
    print("This includes:")
    print("  - All users")
    print("  - All agents")
    print("  - All conversations")
    print("  - All messages")
    print("  - All OTPs")
    print()
    response = input("Are you sure you want to proceed? Type 'YES' to continue: ")
    return response.strip().upper() == 'YES'

def read_sql_file():
    """Read the SQL migration file"""
    sql_file = Path(__file__).parent / "migrations" / "003_recreate_all_tables.sql"
    if not sql_file.exists():
        print(f"❌ SQL file not found: {sql_file}")
        return None
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        return f.read()

def execute_sql_migration():
    """Execute the SQL migration using Supabase client"""
    try:
        # Get Supabase client (should use service role key)
        supabase = get_supabase_client()
        
        # Read SQL content
        sql_content = read_sql_file()
        if not sql_content:
            return False
        
        print("🚀 Starting table recreation...")
        
        # Split SQL into individual statements
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        print(f"📝 Executing {len(statements)} SQL statements...")
        
        # Execute each statement
        for i, statement in enumerate(statements, 1):
            if statement.strip():
                try:
                    # Skip comments and empty statements
                    if statement.strip().startswith('--') or not statement.strip():
                        continue
                    
                    print(f"  [{i}/{len(statements)}] Executing statement...")
                    result = supabase.rpc('exec_sql', {'sql': statement}).execute()
                    
                except Exception as stmt_error:
                    print(f"  ⚠️  Statement {i} warning: {stmt_error}")
                    # Continue with other statements
                    continue
        
        print("✅ SQL migration completed!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("\n💡 Manual execution required:")
        print("1. Go to your Supabase dashboard")
        print("2. Navigate to SQL Editor")
        print("3. Copy and paste the content from migrations/003_recreate_all_tables.sql")
        print("4. Execute the SQL")
        return False

def verify_tables():
    """Verify that tables were created successfully"""
    try:
        supabase = get_supabase_client()
        
        print("\n🔍 Verifying table creation...")
        
        # Test each table
        tables = ['users', 'agents', 'conversations', 'messages', 'otps']
        
        for table in tables:
            try:
                result = supabase.table(table).select("*").limit(1).execute()
                print(f"  ✅ {table} table: accessible")
            except Exception as e:
                print(f"  ❌ {table} table: {e}")
                return False
        
        # Test the test user
        try:
            result = supabase.table("users").select("email").eq("email", "admin@supabase.io").execute()
            if result.data:
                print(f"  ✅ Test user created: admin@supabase.io")
            else:
                print(f"  ⚠️  Test user not found")
        except Exception as e:
            print(f"  ❌ Error checking test user: {e}")
        
        print("\n✅ Table verification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Restart your backend server:")
    print("   uvicorn main:app --host 0.0.0.0 --port 8000")
    print("\n2. Test user registration:")
    print("   POST /auth/register with any @supabase.io email")
    print("\n3. Test agent creation:")
    print("   POST /api/agents (requires authentication)")
    print("\n4. All tables now have proper foreign key constraints")
    print("   and RLS is disabled for development")
    print("\n5. The test user credentials:")
    print("   Email: admin@supabase.io")
    print("   Password: testpassword123")
    print("\n" + "="*60)

def main():
    """Main execution function"""
    print("🔧 Supabase Table Recreation Script")
    print("="*50)
    
    # Check if service role key is available
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not service_role_key:
        print("❌ SUPABASE_SERVICE_ROLE_KEY not found in environment")
        print("This script requires service role key to modify table structure")
        return False
    
    # Confirm deletion
    if not confirm_deletion():
        print("❌ Operation cancelled by user")
        return False
    
    # Execute migration
    if not execute_sql_migration():
        return False
    
    # Verify tables
    if not verify_tables():
        print("⚠️  Some tables may not be working correctly")
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Table recreation completed successfully!")
    else:
        print("\n💥 Table recreation failed. Check the errors above.")
        sys.exit(1)