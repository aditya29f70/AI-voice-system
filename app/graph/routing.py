from typing import Literal

def conversation_should_continue(state) -> Literal['connected', 'disconnected']:
    if state['should_continue']:
        return 'connected'
    else:
        return "disconnected"


def decide_hot_warm(state)-> Literal['hot_action', "warm_action", "__end__"]:
    if not state['actions']['whatsapp_sent_mid_call'] and not state['actions']['callback_scheduled']:
        if state['lead']['confirmed'] and state['lead']['intent']=='hot':
            return 'hot_action'

    if not state['actions']['callback_scheduled'] and not state['actions']['whatsapp_sent_mid_call'] :
        if state['callback']['requested']:
            if state['lead']['confirmed'] and state['lead']['intent']=='warm':
                return "warm_action"

    return "__end__"
