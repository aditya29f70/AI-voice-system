import json

from flask import Flask, Response, request
from flask_sock import Sock


app = Flask(__name__)
sock = Sock(app)


# ============================================================
# TWIML ENDPOINT
# ============================================================

@app.route("/twiml", methods=["POST"])
def twiml():

    print("\n" + "=" * 60)
    print("🌐 TWILIO CALLED /twiml")
    print("=" * 60)

    host = request.host

    stream_url = f"wss://{host}/media"

    print(f"📡 Stream URL: {stream_url}")

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>

    <Start>
        <Stream url="{stream_url}" />
    </Start>

    <Say>
        Hello. Please speak after this message.
    </Say>

    <Pause length="60"/>

</Response>
"""

    return Response(
        xml,
        content_type="text/xml"
    )


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@sock.route("/media")
def media(ws):

    print("\n" + "=" * 60)
    print("🔥 WEBSOCKET /media CONNECTED")
    print("=" * 60)

    while True:

        message = ws.receive()

        if message is None:
            print("🔴 WebSocket closed")
            break

        print("\n📩 MESSAGE RECEIVED")

        try:

            data = json.loads(message)

            event = data.get("event")

            print(f"📡 EVENT: {event}")

            if event == "connected":
                print("🟢 Twilio connected")

            elif event == "start":
                print("🚀 Stream started")

            elif event == "media":
                print("🎤 Audio received")

            elif event == "stop":
                print("🛑 Stream stopped")
                break

        except Exception as e:

            print("❌ Error processing message:")
            print(e)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("🚀 Server starting on port 8000")

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )