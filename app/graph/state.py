from typing import TypedDict, Optional, Literal, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class LeadInfo(TypedDict):
    language: Optional[str]

    what_they_sell: Optional[str]
    budget: Optional[str]
    number_of_products: Optional[int]
    timeline: Optional[str]
    features: list[str]

    intent: Optional[Literal["hot", "warm", "cold"]]


class CallbackInfo(TypedDict):
    requested: bool 
    date: Optional[str]
    time: Optional[str]
    timezone: Optional[str]
    scheduled: bool

class ActionState(TypedDict):
    whatsapp_sent_mid_call: bool
    callback_scheduled: bool
    followup_sent: bool


class VoiceCallState(TypedDict):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Call information
    customer_phone: str 
    my_phone: str 
    call_active: bool

    # Current interaction
    current_audio: Optional[bytes]
    current_transcript: Optional[str]
    current_response: Optional[str]

    # Lead
    lead: LeadInfo

    # Callback
    callback: CallbackInfo

    # Action
    actions: ActionState

    # Control
    should_continue: bool








