import sqlite3

def check_db():
    try:
        conn = sqlite3.connect('chat_app.db')
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in database:", [table[0] for table in tables])
        
        # Check users table structure
        cursor.execute("PRAGMA table_info(users);")
        user_columns = cursor.fetchall()
        print("\nUsers table structure:")
        for col in user_columns:
            print(col)
            
        # Check if any users exist
        cursor.execute("SELECT COUNT(*) FROM users;")
        user_count = cursor.fetchone()[0]
        print(f"\nNumber of users in database: {user_count}")
        
        conn.close()
        print("\nDatabase check completed successfully")
    except Exception as e:
        print(f"Error checking database: {str(e)}")

if __name__ == "__main__":
    check_db() 