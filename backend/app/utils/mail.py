import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "UniHelp")


def send_email(to: str, subject: str, body: str) -> dict:
    if not SMTP_USER or not SMTP_PASSWORD:
        return {"success": False, "error": "SMTP non configure. Verifiez SMTP_USER et SMTP_PASSWORD dans .env"}

    msg = MIMEMultipart("alternative")
    msg["From"] = Header(f"{EMAIL_FROM_NAME} <{EMAIL_FROM}>", "utf-8")
    msg["To"] = Header(to, "utf-8")
    msg["Subject"] = Header(subject, "utf-8")

    html = f"""<html><body style="font-family: Arial, sans-serif; line-height: 1.6;">
<p>{body.replace(chr(10), '<br>')}</p>
<hr>
<p style="color: #888; font-size: 0.85em;">
Cet email a ete genere automatiquement par UniHelp.<br>
IIT & ISB Tunisia
</p>
</body></html>"""

    msg.attach(MIMEText(body, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, to, msg.as_string())
        return {"success": True, "error": None}
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": "Echec d'authentification SMTP. Utilisez un mot de passe d'application Gmail "
                     "(https://myaccount.google.com/apppasswords).",
        }
    except smtplib.SMTPRecipientsRefused:
        return {"success": False, "error": "Destinataire refuse par le serveur SMTP."}
    except Exception as e:
        return {"success": False, "error": f"Erreur SMTP: {e}"}
