import random
import string
import smtplib
import os
import secrets
from datetime import datetime, timedelta, timezone
from jinja2 import Environment, FileSystemLoader
from sqlmodel import select
from src.models import Provider, Student
from src.shared.exceptions import CustomException
from email.mime.text import MIMEText
from src.config import Config
from src.shared.dependency.otp.model import OTP, ResetToken
from src.shared.services.db_service import DatabaseService 

class OTPService:
    def __init__(self):
        self.smtp_server = Config.SMTP_SERVER
        self.smtp_port = Config.SMTP_PORT
        self.email_user = Config.EMAIL_ADDRESS
        self.email_password = Config.EMAIL_APP_PASSWORD
        self.expiry_minutes = Config.OTP_EXPIRY_MINUTES

    def _generate_otp(self, length: int = 6) -> str:
        """Generate a numeric OTP."""
        return ''.join(random.choices(string.digits, k=length))

    async def send_otp(self, db: DatabaseService, recipient_email: str) -> None:
        otp_code = self._generate_otp()
        expiry = datetime.utcnow() + timedelta(minutes=self.expiry_minutes)

        # 🔎 Get the user (Student or Provider)
        user = await db.get_by_field(Student, "email", recipient_email)
        if not user:
            user = await db.get_by_field(Provider, "email", recipient_email)
        if not user:
            raise CustomException(
                dev_message="No account found for this email.",
                user_message="We could not find an account linked to this email."
            )

        # Build salutation
        if hasattr(user, "full_name"):  # Student
            salutation = f"Dear {user.full_name.split()[0]}"
        elif hasattr(user, "type"):  # Provider
            if user.type == "individual" and getattr(user, "first_name", None):
                salutation = f"Dear {user.first_name}"
            elif user.type == "organisation" and getattr(user, "organisation_name", None):
                salutation = f"Dear {user.organisation_name}"
            else:
                salutation = "Dear User"
        else:
            salutation = "Dear User"

        # Clear old OTPs
        stmt = select(OTP).where(OTP.email == recipient_email)
        result = await db.session.scalars(stmt)
        for record in result.all():
            await db.session.delete(record)
        await db.session.commit()

        # Save new OTP
        otp_record = OTP(email=recipient_email, otp=otp_code, expires_at=expiry)
        db.session.add(otp_record)
        await db.session.commit()

        # Load HTML template
        template_dir = os.path.join(os.getcwd(), "src", "shared", "utils")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("email_temp.html")
        email_body = template.render(
            salutation=salutation,
            otp_code=otp_code,
            expiry_minutes=self.expiry_minutes
        )

        # Build email
        msg = MIMEText(email_body, "html")
        msg["Subject"] = "🔐 Your OTP Code for Password Reset"
        msg["From"] = self.email_user
        msg["To"] = recipient_email

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.sendmail(self.email_user, recipient_email, msg.as_string())
        except Exception as e:
            raise CustomException(
                dev_message=f"Failed to send OTP: {e}",
                user_message="Could not send OTP. Please try again later.",
                status_code=500
            )



    async def verify_otp(self, db: DatabaseService, recipient_email: str, otp: str) -> None:
        """
        Verify OTP and generate a ResetToken valid for 10 minutes.
        OTP is deleted after verification.
        """
        # Fetch OTP
        stmt = select(OTP).where(OTP.email == recipient_email)
        result = await db.session.scalars(stmt)
        record = result.one_or_none()

        if not record:
            raise CustomException(
                dev_message="No OTP record found",
                user_message="Invalid OTP. Please request a new one.",
                status_code=400
            )

        # Check expiration
        now = datetime.now(timezone.utc)
        if now > record.expires_at:
            await db.session.delete(record)
            await db.session.commit()
            raise CustomException(
                dev_message="OTP expired",
                user_message="Your OTP has expired. Please request a new one.",
                status_code=400
            )

        # Verify OTP
        if record.otp != otp:
            raise CustomException(
                dev_message="OTP mismatch",
                user_message="Incorrect OTP. Please try again.",
                status_code=400
            )

        # Delete OTP
        await db.session.delete(record)
        await db.session.commit()

        # Generate reset token
        reset_token_str = secrets.token_hex(32)
        expires_at = now + timedelta(minutes=10)

        # save ResetToken
        reset_token_record = ResetToken(
            email=recipient_email,
            token=reset_token_str,
            expires_at=expires_at,
            used=False
        )
        db.session.add(reset_token_record)
        await db.session.commit()


# Global instance (like tokenOperations)
otpOperations = OTPService()

#verify password reset token
async def verify_reset_token(db: DatabaseService, email: str) -> ResetToken:

    """ Verify if a valid reset token exists for the given email.  """

    token_record = await db.get_by_field(ResetToken, "email", email)

    if not token_record or token_record.used:
        raise CustomException(
            dev_message="Invalid reset token",
            user_message="Password reset unsuccessful. You might not have requested for a reset.",
            status_code=400
        )

    now = datetime.now(timezone.utc)
    if now > token_record.expires_at:
        await db.delete_by_field(token_record)
        await db.session.commit()
        raise CustomException(
            dev_message="Reset token expired",
            user_message="Your reset session has expired. Please request for a new OTP.",
            status_code=400
        )

    return token_record