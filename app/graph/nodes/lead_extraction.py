from app.services.llm import llm
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date as DateType
from datetime import time as TimeType
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage

today= datetime.now(
    ZoneInfo("Asia/Kolkata")
).date()

class LeadExtractor(BaseModel):
    what_they_sell: Optional[str]= Field(
        default=None,
        description="What product or type of products the customer sells through their business."
    )
    budget: Optional[str]= Field(
        default=None,
        description="The customer's stated budget or approximate amount they are willing to spend on the project."
    )
    number_of_products: Optional[int]=Field(
        default=None,
        description="The approximate number of products the customer wants to include or sell on the website."
    )
    timeline: Optional[str]= Field(
        default=None,
        description="The customer's expected or preferred timeframe for completing or launching the project."
    )
    features: list[str]=Field(
        default_factory=list,
        description="Specific features or functionalities the customer wants in the website or application."
    )
    intent: Optional[Literal["hot", "warm", "cold"]]= Field(
        default=None,
        description=(
        "The customer's buying intent: 'hot' for strong interest and readiness "
        "to proceed, 'warm' for interest but some uncertainty or delay, and "
        "'cold' for low interest or low likelihood of proceeding."
        )
    )
    confirmed:bool= Field(
        description=(
        "Indicates whether the predicted intent is confirmed by sufficient "
        "evidence from the conversation. True means the LLM has enough "
        "conversation evidence to confidently assign the intent; false means "
        "the intent is still uncertain and should not yet be treated as confirmed."
        )
    )
    requested:bool= Field(
        description="Whether the customer has explicitly requested a callback."
    )
    date: Optional[DateType]= Field(
        default=None,
        description=(
            "The callback date. Convert relative expressions such as "
            "'tomorrow', 'next Monday', etc. into an actual calendar date "
            "using the provided current date."
        )
    )
    time: Optional[TimeType]= Field(
        default=None,
        description=(
            "The preferred callback time. Convert natural language such as "
            "'10 in the morning' into a valid time value."
        )
    )
    timezone: Optional[str]=Field(
        default="Asia/Kolkata",
        description=(
            "The IANA timezone for the callback. Use Asia/Kolkata as the default "
            "timezone unless the customer explicitly specifies a different timezone."
        )
    )


LEAD_PROMPT_TEMPLATE = """
You are a lead extraction assistant for a business that builds websites
and applications.

Current date: {current_date}
Default timezone: Asia/Kolkata

Your task is to analyze the conversation and extract the latest structured
lead and callback information.

The conversation may contain messages from both the customer and the assistant.

### Conversation Language

{language}

The customer may speak in the language specified above. Understand the
conversation regardless of whether it contains English, Hindi, Telugu, or
another language.

### Conversation History

{conversation_history}

### Current Customer Transcript

{current_transcript}

The current transcript is the customer's latest utterance. Treat it as the
most recent information and use it to update or supplement information from
the conversation history.

### Lead Information

Extract the following:

- what_they_sell:
  What product or type of products the customer sells through their business.
  Return null if this information is not available.

- budget:
  The customer's stated or approximate amount they are willing to spend on
  the project. Preserve the currency when available.
  Return null if no budget has been mentioned.

- number_of_products:
  The approximate number of products the customer wants to include or sell.
  Return an integer when possible.
  Return null if it is not mentioned.

- timeline:
  The customer's expected or preferred timeframe for completing or launching
  the project.
  Return null if it is not mentioned.

- features:
  Specific features or functionality requested by the customer.
  Return an empty list if no features have been mentioned.
  Do not invent features.

- intent:
  Classify the customer's buying intent:
    * hot = strong interest and readiness to proceed
    * warm = interested but still evaluating, uncertain, or needs more information
    * cold = low interest or unlikely to proceed

  Return null if there is insufficient evidence to determine the intent.

- confirmed:
  Set true only when there is sufficient evidence in the conversation to
  confidently assign the selected intent.

  Set false when the intent is uncertain or there is insufficient evidence.

  This field refers specifically to confirmation of the predicted lead intent,
  not confirmation of project requirements.

### Callback Information

- requested:
  Set true only if the customer explicitly requests a callback.
  Otherwise set false.

- date:
  The requested callback date.
  Return null if the customer has not provided one.

- time:
  The customer's preferred callback time.
  Return null if it has not been provided.

- timezone:
  The timezone associated with the callback time.
  Do not invent a timezone.
  Return null if it cannot be determined.

### Important Rules

1. Use both the conversation history and current transcript.
2. The current transcript represents the newest customer information.
3. Preserve information that was established earlier in the conversation.
4. If the customer provides new information, update the corresponding field.
5. Do not invent or assume missing information.
6. Do not confuse the assistant's statements with customer-provided information.
7. If information is unavailable, use null for optional fields.
8. For features, use an empty list when no features are known.
9. If no callback has been requested, set requested to false and date, time,
   and timezone to null.
10. Return only the requested structured output.

{format_instructions}
"""


def lead_extraction(state):

    parser= PydanticOutputParser(pydantic_object=LeadExtractor)

    prompt= PromptTemplate(
        template=LEAD_PROMPT_TEMPLATE,
        input_variables=["current_date","language", "conversation_history", "current_transcript"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    conversation_history= "\n".join(f"{message.type}: {message.content}" for message in state['messages'])

    lead_chain= prompt|llm|parser

    lead_output= lead_chain.invoke({"current_date": today, "language":state['language'], "conversation_history": conversation_history, "current_transcript": state['current_transcript']})


    return {
        "lead":{
            "what_they_sell": lead_output.what_they_sell,
            "budget": lead_output.budget,
            "number_of_products": lead_output.number_of_products,
            "timeline": lead_output.timeline,
            "features": lead_output.features,
            "intent": lead_output.intent,
            "confirmed": lead_output.confirmed
        },
        "callback":{
            "requested":lead_output.requested,
            "date": lead_output.date,
            "time": lead_output.time,
            "timezone": lead_output.timezone
        }
    }


test_state={
    "messages":[HumanMessage(content="not confirm right now"), AIMessage(content="so when we can have a next call?"), HumanMessage(content="we can take a call tomorrow? at 9pm")],
    "language":"en",
    "current_transcript":"but I want to build a e-commerace website. I sell cloths, i have around 600 products and my budget is around 60,000 rupees",
}

print(lead_extraction(test_state))





