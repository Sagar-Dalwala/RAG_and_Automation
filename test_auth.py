import requests
import time
import os

# API URL
API_URL = "http://localhost:8000"

def test_auth_flow():
    print("Testing authentication flow...")
    
    # 1. Register a new test user
    try:
        register_response = requests.post(
            f"{API_URL}/auth/register",
            json={"username": "testuser", "password": "testpassword"}
        )
        print(f"Register response: {register_response.status_code}")
        if register_response.status_code not in [200, 400]:  # 400 is ok if user already exists
            print(f"Register failed: {register_response.text}")
            return False
    except Exception as e:
        print(f"Register exception: {str(e)}")
    
    # 2. Login with the test user
    try:
        login_data = {
            "username": "testuser",
            "password": "testpassword"
        }
        login_response = requests.post(
            f"{API_URL}/auth/token",
            data=login_data,
            allow_redirects=False
        )
        print(f"Login response: {login_response.status_code}")
        
        if login_response.status_code != 200:
            print(f"Login failed: {login_response.text}")
            return False
            
        tokens = login_response.json()
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]
        
        # Get cookies
        cookies = login_response.cookies
        print(f"Cookies: {dict(cookies)}")
    except Exception as e:
        print(f"Login exception: {str(e)}")
        return False
    
    # 3. Get current user with access token
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        user_response = requests.get(
            f"{API_URL}/auth/users/me",
            headers=headers,
            cookies=cookies
        )
        print(f"Get user response: {user_response.status_code}")
        
        if user_response.status_code != 200:
            print(f"Get user failed: {user_response.text}")
            return False
            
        print(f"Current user: {user_response.json()}")
    except Exception as e:
        print(f"Get user exception: {str(e)}")
        return False
    
    # 4. Refresh token
    try:
        refresh_data = {"refresh_token": refresh_token}
        refresh_response = requests.post(
            f"{API_URL}/auth/refresh-token",
            json=refresh_data,
            cookies=cookies
        )
        print(f"Refresh token response: {refresh_response.status_code}")
        
        if refresh_response.status_code != 200:
            print(f"Refresh token failed: {refresh_response.text}")
            return False
            
        new_tokens = refresh_response.json()
        new_access_token = new_tokens["access_token"]
        cookies = refresh_response.cookies
    except Exception as e:
        print(f"Refresh token exception: {str(e)}")
        return False
    
    # 5. Get user with new access token
    try:
        headers = {"Authorization": f"Bearer {new_access_token}"}
        user_response = requests.get(
            f"{API_URL}/auth/users/me",
            headers=headers,
            cookies=cookies
        )
        print(f"Get user after refresh: {user_response.status_code}")
        
        if user_response.status_code != 200:
            print(f"Get user after refresh failed: {user_response.text}")
            return False
            
        print(f"Current user after refresh: {user_response.json()}")
    except Exception as e:
        print(f"Get user after refresh exception: {str(e)}")
        return False
    
    # 6. Logout
    try:
        logout_response = requests.post(
            f"{API_URL}/auth/logout",
            cookies=cookies
        )
        print(f"Logout response: {logout_response.status_code}")
        
        if logout_response.status_code != 200:
            print(f"Logout failed: {logout_response.text}")
            return False
    except Exception as e:
        print(f"Logout exception: {str(e)}")
        return False
    
    print("Authentication flow test completed successfully!")
    return True

if __name__ == "__main__":
    # Make sure API is running
    try:
        requests.get(f"{API_URL}")
    except:
        print("API server is not running!")
        exit(1)
        
    # Test auth flow
    test_auth_flow() 