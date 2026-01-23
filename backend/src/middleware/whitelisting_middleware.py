# middleware/ip_whitelist.py

import geoip2.database
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from src.config import Config

class IPWhitelist(BaseHTTPMiddleware):
    """
    Middleware to:
    1. Allow only requests from specific IP addresses or countries.
    2. Use local GeoIP database for country lookup (no external API calls).
    """

    def __init__(self, app, allowed_ips=None, allowed_countries=None, geoip_db_path="GeoLite2-Country.mmdb"):
        super().__init__(app)
        self.allowed_ips = allowed_ips or []  # e.g., ["127.0.0.1", "192.168.0.10"]
        self.allowed_countries = allowed_countries or []  # e.g., ["GH"]
        self.geoip_db_path = geoip_db_path
        self.reader = geoip2.database.Reader(geoip_db_path)

    async def dispatch(self, request: Request, call_next):
        # 1️⃣ Get client IP address
        client_ip = request.client.host #type: ignore

        # 2️⃣ Allow direct IP whitelisting (for trusted addresses)
        if "*" in self.allowed_ips or client_ip in self.allowed_ips:
            return await call_next(request)

        # 3️⃣ Block local/private network IPs (they can't be geolocated)

        if client_ip.startswith("127.") or client_ip.startswith("192.168.") or client_ip.startswith("10."):
            raise HTTPException(status_code=403, detail="Access denied: Local/private network not allowed.")

        # 4️⃣ GeoIP lookup (from local database)
        try:
            response = self.reader.country(client_ip)
            country_code = response.country.iso_code  # e.g., "GH"
            print("=============================")
            print(f"The obtained country code is: {country_code}")
        except:
            country_code = "UNKNOWN"  # IP not found in database

        # 5️⃣ Check if country is allowed
        if self.allowed_countries and country_code not in self.allowed_countries:
            raise HTTPException(status_code=403, detail=f"Access denied: Country '{country_code}' not allowed.")

        # 6️⃣ If passed checks → allow request
        return await call_next(request)

