import os
import smtplib
import asyncio
from pathlib import Path
import mimetypes

from email.message import EmailMessage
from dotenv import load_dotenv


load_dotenv()


class EmailService:

    def __init__(self):

        self.email_address = os.getenv("EMAIL_ADDRESS")
        self.app_password = os.getenv("EMAIL_APP_PASSWORD")


    def _send_email_sync(
        self,
        to_email: str,
        subject: str,
        body: str,
        attachments: list[str] = None
    ):

        message = EmailMessage()

        message["From"] = self.email_address
        message["To"] = to_email
        message["Subject"] = subject

        message.set_content(body)

        if attachments:
            for file_path in attachments:
                path= Path(file_path)

                mime_type, _= mimetypes.guess_type(path)

                if mime_type is None:
                    maintype= "application"
                    subtype= "octet-stream"
                else:
                    maintype, subtype = mime_type.split("/", 1)

                with open(path, "rb") as file:
                    file_data = file.read()

                message.add_attachment(
                    file_data,
                    maintype=maintype,
                    subtype=subtype,
                    filename=path.name
                )

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                self.email_address,
                self.app_password
            )

            smtp.send_message(message)

        return {
            "success": True,
            "to": to_email
        }


    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        attachments: list[str] = None
    ):

        return await asyncio.to_thread(
            self._send_email_sync,
            to_email,
            subject,
            body,
            attachments
        )