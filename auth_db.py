import sqlite3
import hashlib
import os
from datetime import datetime

def init_db():
    # Create database if it doesn't exist
    conn = sqlite3.connect('chat_app.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create chat_history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_provider TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect('chat_app.db')
    c = conn.cursor()
    try:
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                  (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('chat_app.db')
    c = conn.cursor()
    try:
        hashed_password = hash_password(password)
        c.execute('SELECT id FROM users WHERE username = ? AND password = ?',
                  (username, hashed_password))
        user = c.fetchone()
        return user[0] if user else None
    finally:
        conn.close()

def save_chat_history(user_id, query, response, model_name, model_provider):
    conn = sqlite3.connect('chat_app.db')
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO chat_history 
            (user_id, query, response, model_name, model_provider) 
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, query, response, model_name, model_provider))
        conn.commit()
    finally:
        conn.close()

def get_user_chat_history(user_id):
    conn = sqlite3.connect('chat_app.db')
    c = conn.cursor()
    try:
        c.execute('''
            SELECT query, response, model_name, model_provider, created_at 
            FROM chat_history 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        ''', (user_id,))
        return c.fetchall()
    finally:
        conn.close()