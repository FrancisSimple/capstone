import smtplib
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from jinja2 import Template # Import Jinja2

from src.config import Config

class EmailService:
    def __init__(self):
        self.sender = Config.EMAIL_ADDRESS
        self.password = Config.EMAIL_APP_PASSWORD

    def send_verification_email(self, recipient: str, code: str):
        subject = "Verify your account"
        
        # 1. Load the file content
        BASE_DIR = Path(__file__).resolve().parent
        # Make sure to force UTF-8 to avoid that Windows crash again
        template_path = (BASE_DIR / "../utils/email_temp.html").resolve()
        html_content = template_path.read_text(encoding="utf-8")

        # 2. Render the template using Jinja2
        # This replaces {{ otp_code }}, {{ salutation }}, etc. automatically
        template = Template(html_content)
        rendered_html = template.render(
            salutation="Hello there,",  # You can customize this per user if you have their name
            otp_code=code,
            expiry_minutes=5,
            company_name="BinByte Technologies"
        )

        # 3. Construct the Message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = formataddr(("BinByte Security", self.sender)) 
        message["To"] = recipient

        message.attach(MIMEText(rendered_html, "html"))

        # 4. Send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(self.sender, self.password)
            server.sendmail(self.sender, recipient, message.as_string())

email_service = EmailService()