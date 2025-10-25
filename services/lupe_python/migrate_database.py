#!/usr/bin/env python3
"""
Database migration script for Lupe Discord Bot
Adds TV show support columns to existing database
"""

import os
import sys
import psycopg2
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise ValueError("DB_PASSWORD environment variable must be set")

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "cinesync"),
        user=os.getenv("DB_USER", "postgres"),
        password=password
    )

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    cursor.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = %s AND column_name = %s
    """, (table_name, column_name))
    return cursor.fetchone() is not None

def migrate_database():
    """Perform database migration"""
    logger.info("Starting database migration for TV show support...")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Add columns to feedback table
            logger.info("Checking feedback table...")
            
            new_columns = [
                ('content_id', 'INTEGER'),
                ('content_title', 'TEXT'),
                ('content_type', "TEXT DEFAULT 'movie'")
            ]
            
            for column_name, column_type in new_columns:
                if not check_column_exists(cursor, 'feedback', column_name):
                    logger.info(f"Adding column {column_name} to feedback table...")
                    cursor.execute(f"ALTER TABLE feedback ADD COLUMN {column_name} {column_type}")
                else:
                    logger.info(f"Column {column_name} already exists in feedback table")
            
            # 2. Create user_tv_ratings table
            logger.info("Creating user_tv_ratings table...")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_tv_ratings (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    show_id INTEGER NOT NULL,
                    rating REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, show_id)
                )
            ''')
            
            # 3. Create indexes
            logger.info("Creating indexes...")
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_user_tv_ratings_user_id ON user_tv_ratings(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_feedback_content_type ON feedback(content_type)"
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                    logger.info(f"Created index: {index_sql.split()[-1]}")
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")
            
            # 4. Update existing feedback records to have content_type = 'movie'
            if check_column_exists(cursor, 'feedback', 'content_type'):
                logger.info("Updating existing feedback records...")
                cursor.execute("""
                    UPDATE feedback 
                    SET content_type = 'movie', 
                        content_id = movie_id, 
                        content_title = movie_title 
                    WHERE content_type IS NULL AND movie_id IS NOT NULL
                """)
                updated_rows = cursor.rowcount
                logger.info(f"Updated {updated_rows} existing feedback records")
            
            conn.commit()
            logger.info("✅ Database migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False
    
    return True

def verify_migration():
    """Verify that migration was successful"""
    logger.info("Verifying migration...")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check tables exist
            tables_to_check = ['feedback', 'user_ratings', 'user_tv_ratings']
            for table in tables_to_check:
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_name = %s
                """, (table,))
                if cursor.fetchone():
                    logger.info(f"✅ Table {table} exists")
                else:
                    logger.error(f"❌ Table {table} missing")
                    return False
            
            # Check new columns in feedback table
            required_columns = ['content_id', 'content_title', 'content_type']
            for column in required_columns:
                if check_column_exists(cursor, 'feedback', column):
                    logger.info(f"✅ Column feedback.{column} exists")
                else:
                    logger.error(f"❌ Column feedback.{column} missing")
                    return False
            
            logger.info("✅ Migration verification passed!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🗄️ Lupe Database Migration Tool")
    print("=" * 40)
    
    # Check database connection
    try:
        with get_db_connection() as conn:
            logger.info("✅ Database connection successful")
    except Exception as e:
        logger.error(f"❌ Cannot connect to database: {e}")
        logger.error("Please check your database settings in .env file")
        sys.exit(1)
    
    # Perform migration
    if migrate_database():
        if verify_migration():
            print("\n🎉 Migration completed successfully!")
            print("Your database now supports TV show features.")
        else:
            print("\n⚠️ Migration completed but verification failed.")
            print("Please check the logs above.")
    else:
        print("\n💥 Migration failed.")
        print("Please check the error messages above.")
        sys.exit(1)