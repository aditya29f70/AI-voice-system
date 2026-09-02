from app.services.email_service import EmailService


email_service = EmailService()


result = email_service.send_email(
    to_email="adityakumar81raj@gmail.com",
    subject="Test from LangGraph System",
    body="""
Hello!

This is a real test email from my LangGraph voice calling system.

If you received this, the email notification system is working successfully.

Regards,
LangGraph Call System
"""
)

print(result)