import requests
from pydantic import BaseModel
from typing import Optional

GOOGLE_CLIENT_ID = "YOUR_CLIENT_ID_FROM_ENV_OR_CONFIG" # Ideally load from env

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    }
}

def fake_hash_password(password: str):
    return "fakehashed" + password

async def get_current_active_user(token: str):
    # This is a placeholder implementation
    return User(username="test_user", email="test@example.com", disabled=False)

async def verify_google_token(token: str):
    try:
        # Verify the token with Google's API
        response = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={token}")
        if response.status_code != 200:
            return None
        
        id_info = response.json()
        
        # Verify Audience (Optional but recommended)
        # if id_info['aud'] != GOOGLE_CLIENT_ID: return None

        return {
            "username": id_info.get("name"),
            "email": id_info.get("email"),
            "picture": id_info.get("picture"),
            "disabled": False
        }
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None
