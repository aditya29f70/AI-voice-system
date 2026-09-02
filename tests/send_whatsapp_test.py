"""
send_whatsapp_test.py — standalone sanity check for the Twilio WhatsApp
sandbox before wiring it into the mid-call / after-call nodes.

You must have already joined the sandbox from your own WhatsApp first
(send "join <your-code>" to the sandbox number shown in the Twilio Console).

Install: pip install twilio
"""

import os
import json
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID =  os.getenv("ACCOUNT_SID")         # Twilio Console dashboard
AUTH_TOKEN = os.getenv("AUTH_TOKEN")             # Twilio Console dashboard
SANDBOX_NUMBER = os.getenv("SANDBOX_NUMBER")  # the shared sandbox number shown in Console
MY_NUMBER = os.getenv("MY_NUMBER")      # the number you joined the sandbox from

client = Client(ACCOUNT_SID, AUTH_TOKEN)

message = client.messages.create(
    from_=SANDBOX_NUMBER,
    to=MY_NUMBER,
    content_sid= os.getenv("CONTENT_SID"),
    content_variables=json.dumps({
    "1": "hello this for testing purpose",
    "2": "testing purpose"
})
)

print(f"Message sent: {message.sid}, status: {message.status}")