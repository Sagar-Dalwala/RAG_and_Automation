import streamlit as st
import sqlite3
import hashlib
import os
import extra_streamlit_components as stx
from datetime import datetime, timedelta

# Remove duplicate cookie manager initialization
# The cookie manager is already initialized in frontend.py

def get_cookie_manager():
    """Get the singleton cookie manager instance"""
    # Get the cookie manager from session state
    if 'cookie_manager' not in st.session_state:
        st.session_state.cookie_manager = stx.CookieManager(key="unique_cookie_manager")
    return st.session_state.cookie_manager

def init_user_tokens_table():
    """Initialize the user_tokens table if it doesn't exist"""
    conn = sqlite3.connect('chat_app.db')
    c = conn.cursor()
    try:
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
    finally:
        conn.close()

def init_session():
    """Initialize the session with cookie-based authentication"""
    # Initialize user_tokens table
    init_user_tokens_table()
    
    # Get cookie manager
    cookie_manager = get_cookie_manager()
    
    # Always check the token validity on each page load
    # Try to get user_id from cookie
    user_token = cookie_manager.get('user_token')
    if user_token:
        # Verify the token and get user_id
        user_id = verify_token(user_token)
        if user_id:
            st.session_state.user_id = user_id
        else:
            # Invalid or expired token
            st.session_state.user_id = None
            cookie_manager.delete('user_token')
    else:
        # No token found
        st.session_state.user_id = None

def login_user(username, password):
    """Login user and set session cookie"""
    from auth_db import verify_user
    
    # Ensure user_tokens table exists
    init_user_tokens_table()
    
    user_id = verify_user(username, password)
    if user_id:
        # Set session state
        st.session_state.user_id = user_id
        
        # Generate and store token in cookie
        token = generate_token(user_id)
        cookie_manager = get_cookie_manager()
        # Set cookie to expire in 7 days
        expiry = datetime.now() + timedelta(days=7)
        cookie_manager.set('user_token', token, expires_at=expiry)
        
        # Store token in database
        conn = sqlite3.connect('chat_app.db')
        c = conn.cursor()
        try:
            # Delete any existing tokens for this user
            c.execute('DELETE FROM user_tokens WHERE user_id = ?', (user_id,))
            # Store new token
            c.execute('INSERT INTO user_tokens (user_id, token, expires_at) VALUES (?, ?, ?)',
                     (user_id, token, expiry.strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
        finally:
            conn.close()
        
        return True
    return False

def logout_user():
    """Logout user and clear session cookie"""
    # Get user_id before clearing session state
    user_id = st.session_state.user_id
    
    # Clear session state
    st.session_state.user_id = None
    
    # Clear cookie
    cookie_manager = get_cookie_manager()
    token = cookie_manager.get('user_token')
    cookie_manager.delete('user_token')
    
    # Remove token from database
    if token:
        conn = sqlite3.connect('chat_app.db')
        c = conn.cursor()
        try:
            c.execute('DELETE FROM user_tokens WHERE token = ?', (token,))
            conn.commit()
        finally:
            conn.close()

def generate_token(user_id):
    """Generate a secure token for the user"""
    # Create a token with user_id and expiration timestamp
    expiry = datetime.now() + timedelta(days=7)
    expiry_str = expiry.strftime("%Y%m%d%H%M%S")
    # Add a random component for additional security
    random_component = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    token_string = f"{user_id}-{expiry_str}-{random_component}"
    return hashlib.sha256(token_string.encode()).hexdigest()

def verify_token(token):
    """Verify a user token and return user_id if valid"""
    # Check if token is None or empty
    if not token:
        return None
        
    # Ensure user_tokens table exists
    init_user_tokens_table()
    
    # Get all stored tokens from the tokens table
    conn = sqlite3.connect('chat_app.db')
    c = conn.cursor()
    
    try:
        # Check if token exists in the database
        c.execute('SELECT user_id, expires_at FROM user_tokens WHERE token = ?', (token,))
        result = c.fetchone()
        
        if result:
            user_id, expires_at = result
            # Check if token is expired
            expiry_date = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S")
            if expiry_date > datetime.now():
                return user_id
            else:
                # Token expired, remove it
                c.execute('DELETE FROM user_tokens WHERE token = ?', (token,))
                conn.commit()
    finally:
        conn.close()
    
    return None