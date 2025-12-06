import requests
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

# ⬇️ CONFIGURATION (Matches your amplifyconfiguration.json)
AWS_REGION = "ap-southeast-2"
USER_POOL_ID = "ap-southeast-2_CLLyW9heK"
APP_CLIENT_ID = "1s22b43o217js2ivne3nge63vg"

# The URL to download the public keys for signature verification
JWKS_URL = f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{USER_POOL_ID}/.well-known/jwks.json"

# This tells FastAPI to look for the "Authorization: Bearer <token>" header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user_sub(token: str = Depends(oauth2_scheme)) -> str:
    """
    Validates the JWT Token from AWS Cognito.
    Returns: The unique User ID ('sub') if valid.
    Raises: 401 Unauthorized if invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # 1. Download the Public Keys (JWKS) from AWS
        # (In a production app, you might cache this result to reduce network calls)
        response = requests.get(JWKS_URL)
        jwks = response.json()
        
        # 2. Decode the Token Header to find which key was used
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = {}

        # 3. Find the matching key in the JWKS
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"],
                }
        
        if not rsa_key:
            print("Auth Error: Public key not found for this token")
            raise credentials_exception

        # 4. Verify the Token Signature & Claims
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=APP_CLIENT_ID,
            issuer=f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{USER_POOL_ID}"
        )

        # 5. Extract the User ID ('sub')
        user_sub = payload.get("sub")
        if user_sub is None:
            print("Auth Error: Token missing 'sub' claim")
            raise credentials_exception
            
        return user_sub

    except JWTError as e:
        print(f"JWT Verification Failed: {e}")
        raise credentials_exception
    except Exception as e:
        print(f"General Auth Error: {e}")
        raise credentials_exception