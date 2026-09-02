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

# today= datetime.now(
#     ZoneInfo("Asia/Kolkata")
# ).date()

now = datetime.now(ZoneInfo("Asia/Kolkata"))

current_datetime = now.strftime("%Y-%m-%d %H:%M")

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
            "The customer's preferred callback time. "
            "Convert the time to a normalized 24-hour HH:MM format whenever an "
            "exact time can be determined. "
            "Examples: '14:30', '09:00'. "
            "If the customer provides a relative time such as 'in 60 minutes', "
            "calculate the actual callback time using the current date and time. "
            "If an exact time cannot be determined, return null."
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

Your job is to analyze a customer conversation and extract the latest
structured information about the customer's project, buying intent,
and callback request.

Current datetime: {current_datetime}
Default timezone: Asia/Kolkata


========================
CONVERSATION HISTORY
========================

The conversation may contain messages from both the customer and the assistant.

{conversation_history}


========================
CURRENT CUSTOMER TRANSCRIPT
========================

{current_transcript}

The current transcript is the customer's most recent utterance.

Use it together with the conversation history.

If the current transcript provides new information, update the corresponding
field with the newest information.

If information was established earlier and is not contradicted by the latest
transcript, preserve the previously established information.


========================
LEAD INFORMATION
========================

Extract the following information only from statements made or clearly confirmed
by the customer.


- what_they_sell:

  What product, service, business, or type of products the customer sells
  or intends to sell.

  Return null if this information has not been provided by the customer.


- budget:

  The customer's stated, approximate, or estimated budget for the project.

  Preserve the currency when available.

  Examples:
  - "around ₹30,000"
  - "$500 to $1,000"

  Return null if the customer has not mentioned a budget.

  Do not infer a budget from the customer's business size or requirements.


- number_of_products:

  The approximate number of products the customer wants to display or sell.

  Return an integer when a reasonably clear number is provided.

  Return null if no number of products has been mentioned.

  Do not estimate the number.


- timeline:

  The customer's expected or preferred timeframe for completing,
  launching, or starting the project.

  Examples:
  - "within two weeks"
  - "by December"
  - "as soon as possible"

  Return null if no timeline has been mentioned.


- features:

  Specific website or application features explicitly requested or
  clearly discussed by the customer.

  Examples may include:
  - payment integration
  - product filtering
  - user login
  - admin dashboard

  Return an empty list if no features are known.

  Do not invent, assume, or add standard features that the customer
  did not request.


========================
INTENT CLASSIFICATION
========================

Intent must be classified conservatively.

Do not assign an intent merely because the customer is talking about a project.

The allowed values are:

- hot
- warm
- cold
- null


HOT LEAD

Classify the customer as "hot" ONLY when BOTH conditions are satisfied:

1. The customer shows clear interest and readiness to move forward.

   Examples include:
   - explicitly wants to proceed
   - asks how to start
   - asks for the next steps
   - wants to begin the project
   - agrees to move forward
   - requests implementation or a proposal with clear intent to proceed

AND

2. Sufficient project requirements have been collected from the conversation.

   Consider whether the following project information has been discussed:

   - what_they_sell
   - budget(imp)
   - number_of_products
   - timeline
   - features(imp)

Do NOT classify as "hot" simply because some project details were collected.

Do NOT classify as "hot" if the customer is still only exploring options,
asking general questions, or has not shown readiness to proceed.

If the customer is clearly interested but the required project discussion is
still incomplete, do not automatically classify them as hot.



WARM LEAD

Classify the customer as "warm" ONLY when the customer shows interest in
continuing the discussion but is not yet ready to proceed immediately.

In this system, a strong indicator of warm intent is that the customer
explicitly requests a callback or asks to continue the discussion later.

Examples:
- "Call me tomorrow."
- "Can we discuss this later?"
- "I'm busy right now, call me in the evening."
- "Let's talk again next week."

A callback request alone indicates interest in continuing the conversation.

Do not require a complete project specification before identifying a warm lead.

However, if the customer requests a callback but the date or time has not yet
been provided, the callback information should remain incomplete.


-> intent need to be only one hot,warm or cold don't mix these all (until customer is not clear)

COLD LEAD

Classify the customer as "cold" ONLY when there is clear evidence that the
customer is not interested or is unlikely to continue.

Examples:
- explicitly says they are not interested
- rejects the service
- says they do not need a website or application
- clearly asks not to be contacted again

Do not classify a customer as cold merely because they are uncertain,
busy, or have not provided enough information.



========================
CONFIRMED
========================

The "confirmed" field refers ONLY to confidence in the intent classification.

Set confirmed to true ONLY when:

- intent is not null
AND
- there is clear and sufficient evidence in the conversation supporting
  that intent classification.

Set confirmed to false when:

- intent is null
- the evidence is incomplete
- the customer's intent is uncertain
- multiple interpretations are possible


========================
CALLBACK INFORMATION
========================


- requested:

  Set to true ONLY when the customer explicitly requests a callback,
  asks to continue the discussion later, or clearly asks to be contacted
  again at another time.

  Otherwise set to false.


- date:

  Extract the customer's requested callback date.

  Interpret relative dates using the current date:

  Current datetime: {current_datetime}

  Examples:
  - "tomorrow"
  - "next Monday"
  - "on Friday"

  Convert relative dates to an appropriate date when possible.

  Return null if no callback date has been provided.


- time:

  Extract the customer's preferred callback time.

  Examples:
  - "3 PM"
  - "in the evening"
  - "around 10 in the morning"

  Return null if no callback time has been provided.


- timezone:

  Use the timezone explicitly provided by the customer.

  If the customer does not specify a timezone but the context clearly indicates
  the default timezone should apply, use:

  Asia/Kolkata

  Otherwise return null.


========================
IMPORTANT EXTRACTION RULES
========================

1. Use both conversation history and the current transcript.

2. The current transcript is the newest customer information.

3. Preserve valid information established earlier in the conversation.

4. If the customer provides newer information that contradicts previous
   information, use the newest customer-provided information.

5. Extract information only when it comes from the customer or is clearly
   confirmed by the customer.

6. Do not treat suggestions, assumptions, or questions made by the assistant
   as customer requirements.

7. Do not invent missing information.

8. Use null for unknown optional fields.

9. Use an empty list [] when no features are known.

10. If no callback has been requested:

    - requested = false
    - date = null
    - time = null
    - timezone = null

11. If a callback is requested but date or time is missing, preserve:

    - requested = true

    and set only the missing fields to null.

12. Intent classification must be conservative.

    When uncertain, use:

    - intent = null
    - confirmed = false

13. Return only the requested structured output.

{format_instructions}
"""


async def lead_extraction(state):

    parser= PydanticOutputParser(pydantic_object=LeadExtractor)

    prompt= PromptTemplate(
        template=LEAD_PROMPT_TEMPLATE,
        input_variables=["current_datetime", "conversation_history", "current_transcript"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    conversation_history= "\n".join(f"{message.type}: {message.content}" for message in state['messages'])

    lead_chain= prompt|llm|parser

    lead_output= await lead_chain.ainvoke({"current_datetime": current_datetime, "conversation_history": conversation_history, "current_transcript": state['current_transcript']})


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
            "timezone": lead_output.timezone,
            "callback_situation": state['callback']['callback_situation']
        },
        "actions": state['actions']
        
    }


# test_state={
#     "messages":[HumanMessage(content="not confirm right now"), AIMessage(content="so when we can have a next call?"), HumanMessage(content="we can take a call tomorrow? at 9pm")],
#     "language":"en",
#     "current_transcript":"but I want to build a e-commerace website. I sell cloths, i have around 600 products and my budget is around 60,000 rupees",
# }

# print(lead_extraction(test_state))





