from app.services.whatsapp_service import WhatsAppService

whatsapp= WhatsAppService()

def hot_action(state):
    customer_phone= state['customer_phone']

    message= (
        "Hi!"
        "Thank you for your interest."
        "We've noted your requirements and our team will assist you further."
    )

    try:
        # result= await whatsapp.send_text_message(
        #     phone_number=customer_phone,
        #     message=message
        # )

        print(message)


        return {"actions":{"whatsapp_sent_mid_call":True, "callback_scheduled": state['actions']['callback_scheduled']}, "lead":state['lead'], "callback": state['callback']}

    except Exception as e:
        print("WhatsApp sending failed: {e}")

        return {
            "actions": {"whatsapp_sent_mid_call":False,"callback_scheduled": state['actions']['callback_scheduled'],"lead":state['lead'], "callback": state['callback']}
        }