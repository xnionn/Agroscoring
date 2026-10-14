import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def send_email_html(to_email: str, subject: str, html_content: str) -> tuple[bool, str]:
    """
    Sends an HTML email using SMTP configuration from environment variables.
    Returns (success_boolean, message_string)
    """
    # Load configuration
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = os.environ.get("SMTP_PORT", "465")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        return False, "Настройки SMTP не заданы в .env файле"

    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        # Attach HTML content
        part = MIMEText(html_content, "html", "utf-8")
        msg.attach(part)

        # Connect and send
        # Using SMTP_SSL for port 465 (default for Gmail SSL)
        port = int(smtp_port)
        if port == 465:
            with smtplib.SMTP_SSL(smtp_host, port) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, to_email, msg.as_string())
        else:
            # Fallback for STARTTLS (e.g. port 587)
            with smtplib.SMTP(smtp_host, port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, to_email, msg.as_string())

        return True, "Письмо успешно отправлено"
    except Exception as e:
        return False, f"Ошибка при отправке письма: {str(e)}"
