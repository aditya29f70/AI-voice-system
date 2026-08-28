"""
place_call.py — triggers an outbound call from Twilio to your own (verified)
number, for testing before you touch the real target number.

Run this AFTER media_stream_server.py is running and ngrok is pointed at it.

Install: pip install twilio
"""
import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")                    # Twilio Console dashboard
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")                     # Twilio Console dashboard
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")                       # the trial number Twilio gave you
MY_NUMBER = os.getenv("MY_NUMBER")                         # your own verified number
NGROK_URL = "https://bobcat-illusive-ripening.ngrok-free.dev"    # update each time ngrok restarts

client = Client(ACCOUNT_SID, AUTH_TOKEN)

call = client.calls.create(
    to=MY_NUMBER,
    from_=TWILIO_NUMBER,
    url=f"{NGROK_URL}/twiml",  # Twilio fetches call instructions from here
)

print(f"Call started: {call.sid}")