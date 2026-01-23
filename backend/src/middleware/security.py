from fastapi import Request, Response
from src.config import Config

async def security_headers_middleware(request: Request, call_next):
    response: Response = await call_next(request)
    
    # Always-on security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    
    # Content Security Policy (CSP)
    csp_policy = (
        "default-src 'self'; " # By default, only allow content (scripts, images, styles) from my own server.
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; " # allow javascript scripts from only my site
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; " # allow css from my site and google
        "img-src *" # Images can come from anywhere
    )
    response.headers["Content-Security-Policy"] = csp_policy
    
    # Production-only headers
    try:
        if Config.PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        else:
            print("we are not in production")
    except Exception as e:
        print("An error occurred:", e)   
    
    return response
