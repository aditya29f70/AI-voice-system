from typing import Literal

def conversation_should_continue(state) -> Literal['connected', 'disconnected']:
    if state['should_continue']:
        return 'connected'
    else:
        return "disconnected"


def decide_hot_warm(state)-> Literal['hot_action', "warm_action", "__end__"]:
    if not state['actions']['whatsapp_sent_mid_call']:
        if state['lead']['confirmed'] and state['lead']['intent']=='hot':
            if state['lead']['budget'] and state['lead']['number_of_products'] and state['lead']['features']:
                return 'hot_action'

    if not state['actions']['callback_scheduled']:
        if state['callback']['requested']:
            if state['lead']['confirmed'] and state['lead']['intent']=='warm':
                if state['callback']['date'] and state['callback']['time']:
                    return "warm_action"

    return "__end__"
