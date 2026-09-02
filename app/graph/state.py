from typing import TypedDict, Optional, Literal, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class LeadInfo(TypedDict):

    what_they_sell: Optional[str]
    budget: Optional[str]
    number_of_products: Optional[int]
    timeline: Optional[str]
    features: list[str]

    confirmed: bool
    intent: Optional[Literal["hot", "warm", "cold"]]


class CallbackInfo(TypedDict):
    requested: bool 
    date: Optional[str]
    time: Optional[str]
    timezone: Optional[str]
    callback_situation: Optional[str]

class ActionState(TypedDict):
    whatsapp_sent_mid_call: bool
    callback_scheduled: bool



class VoiceCallState(TypedDict):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    language: Optional[str]

    # Call information
    customer_phone: str 
    my_phone: str 
    # call_active: bool

    # Current interaction
    current_audio: Optional[bytes]
    current_transcript: Optional[str]
    current_response: Optional[str]

    # # Lead
    # lead: LeadInfo

    # # Callback
    # callback: CallbackInfo

    callback_situation: Optional[str]

    # # Action
    # actions: ActionState

    # Control
    should_continue: bool


class LeadExecutionState(TypedDict):
    # Conversation
    messages: list[BaseMessage]
    thread_id: Optional[str]

    customer_id: Optional[int]
    customer_phone: str 
    my_phone: str 
    customer_email: Optional[str]

    current_transcript: Optional[str]
    current_response: Optional[str]

    # Lead
    lead: LeadInfo

    # Callback
    callback: CallbackInfo

    # Action
    actions: ActionState







