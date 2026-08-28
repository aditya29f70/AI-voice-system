import websocket

url = "wss://bobcat-illusive-ripening.ngrok-free.dev/media"

print(f"Connecting to: {url}")

try:
    ws = websocket.create_connection(url)

    print("✅ WEBSOCKET CONNECTION SUCCESSFUL!")

    input("Press Enter to close...")

    ws.close()

except Exception as e:
    print("❌ CONNECTION FAILED")
    print(type(e).__name__)
    print(e)