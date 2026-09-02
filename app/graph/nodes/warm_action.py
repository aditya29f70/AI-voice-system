

def warm_action(state):

    callback_situation=""
    if not state['callback']['date']:
        callback_situation+= "date of callback is not provided,"

    if not state['callback']['time']:
        callback_situation+= "time to callback is not provided"

    if callback_situation:
        return {"callback":{"callback_situation": callback_situation}}

    # For now, simulate scheduling
    print("Scheduling callback...")
    print(f"Date: {state['callback']['date']}")
    print(f"Time: {state['callback']['time']}")
    print(f"Timezone: {state['callback']['timezone']}")

    return {"actions":{"callback_scheduled":True, "whatsapp_sent_mid_call": state['actions']['whatsapp_sent_mid_call']}, "lead":state['lead'], "callback": state['callback']}