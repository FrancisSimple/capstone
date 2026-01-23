from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Initialize the hasher once. 
# It automatically handles salt generation and security parameters.
ph = PasswordHasher()

def generate_password_hash(password: str) -> str:
    """
    Generate a hashed password using Argon2id.
    """
    # Unlike bcrypt, we don't need to manually encode to bytes or generate salt.
    # The library handles that internally.
    return ph.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    Returns True if the passwords match, False otherwise.
    """
    try:
        # Argon2's verify method returns True on success, 
        # or raises a VerifyMismatchError if they don't match.
        return ph.verify(hashed_password, plain_password)
    except VerifyMismatchError:
        return False