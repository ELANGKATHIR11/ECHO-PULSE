import os
import base64
from typing import Optional
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Secret Key Management (Stored securely in local app/system env or generated deterministic edge master key)
_STATIC_SALT = b"EchoPulseNet_Marine_Edge_Security_2026_Postgres_Key_32B!"
_DEFAULT_KEY = base64.urlsafe_b64encode(_STATIC_SALT[:32])

def get_fernet_cipher() -> Fernet:
    secret_key = os.getenv("ECHOPULSENET_SECRET_KEY")
    if not secret_key:
        fernet_key = _DEFAULT_KEY
    else:
        # Pad or derive standard 32 bytes url-safe base64
        padded = (secret_key + "================================")[:32].encode("utf-8")
        fernet_key = base64.urlsafe_b64encode(padded)
    return Fernet(fernet_key)

def encrypt_credential(raw_text: str) -> str:
    """Encrypts plaintext credentials to an opaque Fernet token."""
    cipher = get_fernet_cipher()
    encrypted_bytes = cipher.encrypt(raw_text.encode("utf-8"))
    return encrypted_bytes.decode("utf-8")

def decrypt_credential(encrypted_token: str) -> str:
    """Decrypts opaque Fernet token back to plaintext."""
    if not encrypted_token:
        return ""
    try:
        cipher = get_fernet_cipher()
        decrypted_bytes = cipher.decrypt(encrypted_token.encode("utf-8"))
        return decrypted_bytes.decode("utf-8")
    except Exception as e:
        # If already plaintext or invalid token, return as is
        return encrypted_token

def resolve_db_connection_url() -> str:
    """
    Safely resolves the database URL from encrypted credentials or fallback encrypted connection string.
    """
    encrypted_db_url = os.getenv("ENCRYPTED_DATABASE_URL")
    if encrypted_db_url:
        return decrypt_credential(encrypted_db_url)

    enc_user = os.getenv("ENC_POSTGRES_USER")
    enc_pass = os.getenv("ENC_POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "echopulse_postgis")

    if enc_user and enc_pass:
        user = decrypt_credential(enc_user)
        pwd = decrypt_credential(enc_pass)
        import urllib.parse
        encoded_pwd = urllib.parse.quote_plus(pwd)
        return f"postgresql://{user}:{encoded_pwd}@{host}:{port}/{db}"

    # Fallback to standard env variables if provided
    return os.getenv(
        "POSTGIS_DATABASE_URL",
        os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/echopulse_postgis")
    )
