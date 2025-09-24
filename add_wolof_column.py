#!/usr/bin/env python3
"""
Migration script to add answer_wolof column to messages table
"""
import sqlite3
import sys
from pathlib import Path

def add_wolof_column():
    """Add answer_wolof column to messages table"""
    db_path = "data/jaari_rag.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(messages)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'answer_wolof' in columns:
            print("✅ Column 'answer_wolof' already exists")
            return True
        
        # Add the new column
        print("🔄 Adding 'answer_wolof' column to messages table...")
        cursor.execute("ALTER TABLE messages ADD COLUMN answer_wolof TEXT")
        
        # Commit changes
        conn.commit()
        print("✅ Successfully added 'answer_wolof' column")
        
        # Verify the column was added
        cursor.execute("PRAGMA table_info(messages)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'answer_wolof' in columns:
            print("✅ Column verification successful")
            
            # Show updated schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'")
            schema = cursor.fetchone()[0]
            print(f"\n📋 Updated schema:\n{schema}")
            
            return True
        else:
            print("❌ Column verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Error adding column: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    success = add_wolof_column()
    sys.exit(0 if success else 1)
