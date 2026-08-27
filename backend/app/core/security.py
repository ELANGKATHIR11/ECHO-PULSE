import os
import base64
from typing import Optional
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Load env variables
load_dotenv()

import os
import sys
import base64
import secrets
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Secure key storage path on Jetson / Linux OS (Restricted permissions)
KEY_STORAGE_DIR = Path(os.getenv("ECHOPULSENET_CONFIG_DIR", "/etc/echopulse" if sys.platform != "win32" else os.path.expanduser("~/.echopulse")))
KEY_FILE_PATH = KEY_STORAGE_DIR / ".echopulse_master.key"

def _ensure_runtime_key() -> bytes:
    """
    Safely retrieves or generates a strong 256-bit Fernet key outside source code.
    Fails safely if permissions or environment are compromised.
    """
    env_key = os.getenv("ECHOPULSENET_SECRET_KEY")
    if env_key:
        # Standardize 32-byte url-safe base64
        padded = (env_key + "================================")[:32].encode("utf-8")
        return base64.urlsafe_b64encode(padded)

    # Check local key storage file
    if KEY_FILE_PATH.exists():
        try:
            with open(KEY_FILE_PATH, "rb") as f:
                key = f.read().strip()
                if len(key) == 44:  # Valid base64 Fernet key length
                    return key
        except Exception:
            pass

    # Generate new strong random Fernet key at runtime
    new_key = Fernet.generate_key()
    try:
        KEY_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            os.chmod(KEY_STORAGE_DIR, 0o700)
        with open(KEY_FILE_PATH, "wb") as f:
            f.write(new_key)
        if sys.platform != "win32":
            os.chmod(KEY_FILE_PATH, 0o600)
    except Exception:
        # If filesystem write restricted (e.g. read-only container), key lives in process memory
        pass

    return new_key

def get_fernet_cipher() -> Fernet:
    key = _ensure_runtime_key()
    return Fernet(key)

def encrypt_credential(raw_text: str) -> str:
    """Encrypts plaintext credentials to an opaque Fernet token."""
    if not raw_text:
        return ""
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
    except Exception:
        # If decryption fails or token is invalid, fail safely without leaking
        return ""

def resolve_db_connection_url() -> str:
    """
    Safely resolves the database URL from encrypted credentials or secure environment.
    Never exposes hardcoded fallback passwords.
    """
    encrypted_db_url = os.getenv("ENCRYPTED_DATABASE_URL")
    if encrypted_db_url:
        dec = decrypt_credential(encrypted_db_url)
        if dec:
            return dec

    enc_user = os.getenv("ENC_POSTGRES_USER")
    enc_pass = os.getenv("ENC_POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "echopulse_gis")

    if enc_user and enc_pass:
        user = decrypt_credential(enc_user)
        pwd = decrypt_credential(enc_pass)
        if user and pwd:
            import urllib.parse
            encoded_pwd = urllib.parse.quote_plus(pwd)
            return f"postgresql://{user}:{encoded_pwd}@{host}:{port}/{db}"

    # Return environment-configured DB URL (without hardcoded passwords)
    return os.getenv("DATABASE_URL", os.getenv("POSTGIS_DATABASE_URL", ""))
