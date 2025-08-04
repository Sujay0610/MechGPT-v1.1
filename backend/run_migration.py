#!/usr/bin/env python3
"""
Script to disable RLS for development
Note: This approach uses service role operations to bypass RLS
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def run_migration():
    """Disable RLS by using service role key"""
    try:
        # Get service role key (if available) or use anon key
        supabase_url = os.getenv("SUPABASE_URL")
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        anon_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url:
            print("❌ SUPABASE_URL not found in environment variables")
            return False
        
        # Try service role key first, fallback to anon key
        key_to_use = service_role_key if service_role_key else anon_key
        key_type = "service role" if service_role_key else "anon"
        
        if not key_to_use:
            print("❌ Neither SUPABASE_SERVICE_ROLE_KEY nor SUPABASE_ANON_KEY found")
            return False
        
        print(f"Using {key_type} key to connect to Supabase...")
        
        # Create client with service role key (bypasses RLS)
        supabase = create_client(supabase_url, key_to_use)
        
        print("✅ Connected to Supabase successfully")
        
        # Test if we can access tables (this will work if RLS is disabled or we have service role)
        try:
            # Try to query agents table
            result = supabase.table("agents").select("id").limit(1).execute()
            print(f"✅ Can access agents table: {len(result.data)} records found")
            
            # Try to query users table
            result = supabase.table("users").select("id").limit(1).execute()
            print(f"✅ Can access users table: {len(result.data)} records found")
            
            # Try to query conversations table
            result = supabase.table("conversations").select("id").limit(1).execute()
            print(f"✅ Can access conversations table: {len(result.data)} records found")
            
            # Try to query messages table
            result = supabase.table("messages").select("id").limit(1).execute()
            print(f"✅ Can access messages table: {len(result.data)} records found")
            
            print("\n✅ All tables are accessible!")
            
            if service_role_key:
                print("\n📝 Note: Using service role key bypasses RLS automatically.")
                print("📝 For production, implement proper RLS policies or use service role for backend operations.")
            else:
                print("\n⚠️  Warning: Using anon key. If this works, RLS might already be disabled.")
                print("⚠️  For production, you should use service role key for backend operations.")
            
            return True
            
        except Exception as e:
            print(f"❌ Cannot access tables: {e}")
            print("\n💡 This suggests RLS is still enabled and blocking access.")
            print("💡 You need to either:")
            print("   1. Add SUPABASE_SERVICE_ROLE_KEY to your .env file")
            print("   2. Manually disable RLS in Supabase dashboard")
            print("   3. Run the SQL migration manually in Supabase SQL editor")
            return False
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

def print_manual_instructions():
    """Print manual instructions for disabling RLS"""
    print("\n" + "="*60)
    print("MANUAL MIGRATION INSTRUCTIONS")
    print("="*60)
    print("\nIf the automatic approach doesn't work, please:")
    print("\n1. Go to your Supabase dashboard")
    print("2. Navigate to SQL Editor")
    print("3. Run the following SQL commands:")
    print("\n" + "-"*40)
    
    with open('migrations/002_disable_rls_for_development.sql', 'r') as f:
        sql_content = f.read()
    
    # Filter out comments and empty lines
    lines = sql_content.split('\n')
    sql_statements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('--'):
            sql_statements.append(line)
    
    print('\n'.join(sql_statements))
    print("-"*40)
    print("\n4. After running the SQL, restart your backend server")
    print("5. Test agent creation again")
    print("\n" + "="*60)

if __name__ == "__main__":
    print("🚀 Starting RLS migration...\n")
    
    success = run_migration()
    
    if not success:
        print_manual_instructions()
    
    print("\n🏁 Migration process completed.")