#!/usr/bin/env python3
"""
Migration script to add personalization tables to CineSync v2 database

This script adds the following tables:
- user_embeddings: Vector representations of user preferences
- user_preferences: Analyzed preference patterns
- user_model_weights: Per-user model ensemble weights

Usage:
    python migrate_personalization.py
"""

import os
import sys
import psycopg2
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, using environment variables only")


def get_db_connection():
    """Get database connection from environment variables"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'cinesync'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Error connecting to database: {e}")
        print("\nPlease check your database connection settings:")
        print(f"  Host: {os.getenv('DB_HOST', 'localhost')}")
        print(f"  Port: {os.getenv('DB_PORT', '5432')}")
        print(f"  Database: {os.getenv('DB_NAME', 'cinesync')}")
        print(f"  User: {os.getenv('DB_USER', 'postgres')}")
        sys.exit(1)


def check_existing_tables(cursor):
    """Check which personalization tables already exist"""
    tables_to_check = ['user_embeddings', 'user_preferences', 'user_model_weights']
    existing_tables = []

    for table in tables_to_check:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            );
        """, (table,))

        if cursor.fetchone()[0]:
            existing_tables.append(table)

    return existing_tables


def migrate():
    """Run migration to add personalization tables"""
    print("🚀 CineSync v2 Personalization Migration")
    print("=" * 60)

    # Check if SQL file exists
    sql_file = Path(__file__).parent / 'database_extensions.sql'
    if not sql_file.exists():
        print(f"❌ Error: SQL file not found at {sql_file}")
        sys.exit(1)

    print(f"📄 Using SQL file: {sql_file}")

    # Connect to database
    print("\n🔌 Connecting to database...")
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check database connection
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print(f"✅ Connected to PostgreSQL: {version.split(',')[0]}")

    # Check existing tables
    print("\n🔍 Checking existing tables...")
    existing_tables = check_existing_tables(cursor)

    if existing_tables:
        print(f"⚠️  Found existing tables: {', '.join(existing_tables)}")
        print("   These will be preserved (using IF NOT EXISTS clauses)")
    else:
        print("✅ No existing personalization tables found")

    # Read and execute SQL file
    print("\n📝 Reading migration SQL...")
    with open(sql_file, 'r') as f:
        sql = f.read()

    print("🔨 Executing migration...")
    try:
        cursor.execute(sql)
        conn.commit()
        print("✅ Migration SQL executed successfully")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"❌ Error executing migration: {e}")
        cursor.close()
        conn.close()
        sys.exit(1)

    # Verify tables were created
    print("\n🔍 Verifying tables...")
    new_tables = check_existing_tables(cursor)

    for table in ['user_embeddings', 'user_preferences', 'user_model_weights']:
        if table in new_tables:
            cursor.execute(f"""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_name = %s;
            """, (table,))
            col_count = cursor.fetchone()[0]
            print(f"  ✅ {table}: {col_count} columns")
        else:
            print(f"  ❌ {table}: NOT FOUND")

    # Check indexes
    print("\n🔍 Verifying indexes...")
    cursor.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE tablename IN ('user_embeddings', 'user_preferences', 'user_model_weights')
        ORDER BY indexname;
    """)

    indexes = cursor.fetchall()
    for idx in indexes:
        print(f"  ✅ {idx[0]}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Migration completed successfully!")
    print("\n📊 Summary:")
    print(f"  - Tables created/verified: {len(new_tables)}")
    print(f"  - Indexes created: {len(indexes)}")
    print("\n🎯 Next steps:")
    print("  1. Run test script: python test_personalization.py")
    print("  2. Start the Discord bot with personalization enabled")
    print("  3. Test with /my_recommendations command")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    try:
        migrate()
    except KeyboardInterrupt:
        print("\n\n❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
