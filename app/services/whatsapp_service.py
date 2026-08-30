import os
import httpx


class WhatsAppService:

    def __init__(self):

        self.access_token= os.getenv("WHATSAPP_ACCESS_TOKEN")   
        self.phone_number_id= os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        self.url=(
            f"https://graph.facebook.com/",
            f"vXX.X/{self.phone_number_id}/messages"
        )


    async def send_text_message(
            self,
            phone_number:str,
            message: str
    ):
        headers={
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        playload={
            "messaging_product": "whatsapp",
            "to":phone_number,
            "type": "text",
            "text":{
                "body":message
            }
        }

        async with httpx.AsyncClient() as client:

            response= await client.post(
                self.url,
                headers=headers,
                json= playload
            )

            response.raise_for_status()

            return response.json()