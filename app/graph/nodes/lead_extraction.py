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
You are the background lead-analysis system for a business that builds
websites and applications.

Your job is to maintain the customer's latest structured project information,
buying intent, and callback request.

You are NOT responsible for generating conversational responses.

Current datetime: {current_datetime}
Default timezone: Asia/Kolkata

========================
CONVERSATION
============

Conversation history:

{conversation_history}

Latest customer transcript:

{current_transcript}

The current transcript is the newest customer information.

Use both the history and the latest transcript.

Preserve previously established information unless the customer provides
new or contradictory information.

Only treat information as customer information when the customer states it
or clearly confirms it.

========================
PROJECT INFORMATION
===================

Extract:

* what_they_sell:
  What product, service, business, or type of products the customer sells
  or intends to sell.

* budget:
  The customer's stated or approximate project budget.
  Preserve the currency when available.
  Do not infer a budget.

* number_of_products:
  Approximate number of products the customer wants to display or sell.
  Return an integer only when reasonably clear.
  Do not estimate.

* timeline:
  The customer's expected timeframe for completing, launching, or starting
  the project.

* features:
  Specific website/application features explicitly requested or confirmed
  by the customer.
  Do not add standard or assumed features.

Use null for unknown scalar fields and [] when no features are known.

========================
LEAD INTENT
===========

The lead intent represents the customer's BUYING INTENT.

Allowed values:

* hot
* warm
* cold
* null

IMPORTANT:

Intent and callback are SEPARATE concepts.

A callback request does NOT automatically make a lead warm.

A customer can be:

* HOT + callback requested
* HOT + no callback requested
* WARM + callback requested
* WARM + no callback requested
* COLD + callback requested

Do not use callback status as the only reason to change lead intent.

---

## HOT

Classify as **HOT** when the customer shows **strong intent or readiness to move toward getting the project built**, and there is enough project context to understand what they need.

HOT can be based on either **explicit or strong implicit buying intent**.

Strong HOT signals include:

* explicitly wants to proceed
* wants to start the project
* asks how to begin
* asks for next steps
* agrees to move forward
* asks for implementation
* requests a proposal or quotation **and shows intent to move forward**
* clearly indicates they want the business to build the project
* provides detailed project requirements and actively engages in planning the project
* discusses practical implementation details such as features, products, timeline, branding, payment setup, or other project requirements
* agrees to receive an estimate, proposal, or other next-step business information
* indicates they want the project built soon or within a defined timeframe

### Sufficient project context

The customer does NOT need to provide every field.

Sufficient context may include some combination of:

* what they sell/do
* what they want to build
* important features
* number of products, when relevant
* budget
* timeline
* design/branding requirements
* technical or business requirements

**Budget is NOT required for HOT.**

**Number of products is NOT required for HOT.**

**Every feature is NOT required for HOT.**

Do not downgrade a customer from HOT merely because some information is still unknown.

Do not classify as HOT only because the customer is talking about a project.

Do not classify as HOT when the customer is only:

* researching
* comparing options
* asking general questions
* casually exploring the possibility

unless their overall behavior also shows strong intent to move toward getting the project built.

---

## WARM

Classify as **WARM** when the customer shows genuine interest but there is **not yet enough evidence of current readiness or strong intent to move toward the project**.

Examples:

* exploring the possibility of a website/application
* asking about services without showing commitment
* asking about pricing while still evaluating options
* interested but still comparing alternatives
* wants to think about it
* says they may do it later
* wants to discuss the project later
* interested but not currently ready to proceed
* provides some project requirements but remains exploratory or non-committal

Missing information alone does NOT make a lead WARM.

A customer can be HOT even if their budget, exact requirements, or other fields are unknown.

A callback request is a strong signal of continued interest, but **callback status and intent are independent**.

If a customer is already HOT and then requests a callback:

```text
intent = hot
callback.requested = true
```

Do NOT downgrade HOT to WARM merely because:

* the customer wants to talk later
* the customer requests a callback
* the customer needs additional information
* the customer asks questions
* the customer has not provided a budget
* some requirements are still unknown

---

## COLD

Classify as **COLD only when there is clear evidence that the customer does not want the service or does not want to continue the sales conversation.**

Examples:

* explicitly says they are not interested
* rejects the service
* says they do not need a website/application
* clearly says they do not want to continue
* clearly asks not to be contacted again
* explicitly declines the proposed service

Do NOT classify as COLD merely because:

* information is missing
* the customer is uncertain
* the customer is busy
* the customer needs time
* the customer wants to think about it
* the customer requests a callback
* the customer does not provide a budget
* the customer does not immediately commit

---

## IMPORTANT CLASSIFICATION RULES

### 1. Intent is about buying readiness, not information completeness

Do not use:

```text
missing budget → WARM
missing feature → WARM
missing product count → WARM
```

Instead evaluate the customer's **overall buying behavior**.

### 2. HOT should be sticky

Once a customer has clearly demonstrated HOT intent, do not downgrade them simply because the conversation continues or because additional information is collected.

Only change HOT → WARM if there is **new evidence that their actual buying intent has decreased**.

### 3. Callback does not determine intent

Callback is a separate state.

```text
HOT + callback requested → HOT
WARM + callback requested → WARM
```

Do not use callback alone to determine intent.

### 4. Actions must not influence intent

Do not change intent because an email, WhatsApp message, proposal, or callback was sent/scheduled.

The intent represents the **customer's intent**, not the action taken by the system.

### 5. Use the whole conversation

Determine intent from the **entire conversation**, not only the customer's latest message.

A customer who gradually provides detailed requirements and moves toward an estimate can be HOT even if their latest message is only:

> "That's everything for now."

### 6. Confidence

`confirmed = true` means the classifier is confident about the customer's current intent classification.

It does NOT mean that the customer explicitly confirmed every project requirement.


========================
INTENT STABILITY
================

Intent should be STABLE across turns.

Do not change an already established intent without new evidence from
the customer.

If the current customer message does not clearly change the customer's
buying intent, PRESERVE the previous intent.

Especially:

HOT must remain HOT unless the customer clearly indicates reduced,
withdrawn, or changed buying intent.

Do NOT change:

HOT → WARM

simply because:

* the conversation continues
* more information is being collected
* some project fields are still missing
* the customer asks a question
* the customer requests a callback
* the customer needs time to discuss details

Similarly, do not change WARM → COLD without clear evidence of rejection
or lack of interest.

When intent cannot yet be determined:

intent = null
confirmed = false

Do not force a classification.

========================
CONFIRMED
=========

"confirmed" represents confidence in the CURRENT intent classification.

Set confirmed = true only when:

* intent is hot, warm, or cold
* there is clear evidence supporting that classification
* there is no major ambiguity

Set confirmed = false when:

* intent is null
* evidence is weak
* the customer's intent is ambiguous
* multiple interpretations are possible

Do not use confirmed to mean that the customer confirmed their project
requirements.

========================
CALLBACK
========

Callback is an independent state.

Set requested = true ONLY when the customer explicitly:

* asks for a callback
* asks to be contacted later
* asks to continue the discussion later
* gives a clear request for another call

Do NOT infer callback intent from:

* HOT intent
* WARM intent
* project interest
* budget
* timeline
* missing information
* the customer being busy unless they explicitly ask to be contacted later

If no callback is requested:

requested = false
date = null
time = null
timezone = null

---

## CALLBACK DATE

Extract the requested callback date.

Interpret relative dates using:

Current datetime: {current_datetime}

Examples:

* tomorrow
* next Monday
* Friday
* next week

Convert to an appropriate date when possible.

Return null when no callback date is provided.

---

## CALLBACK TIME

Extract the customer's requested callback time.

Examples:

* 3 PM
* around 10 AM
* in the evening
* after lunch

Return null when no callback time is provided.

---

## CALLBACK TIMEZONE

Use the timezone explicitly provided by the customer.

If no timezone is provided and the conversation clearly uses the default
timezone, use:

Asia/Kolkata

Otherwise return null.

========================
IMPORTANT ACTION SEPARATION
===========================

The lead model determines:

* project information
* lead intent
* callback request

The action system determines what to DO with that information.

Therefore:

DO NOT classify a lead differently because an action needs to happen.

DO NOT create callback intent because an email action is required.

DO NOT create callback intent because the lead is HOT.

DO NOT downgrade HOT because the HOT action has already happened.

DO NOT decide whether an email, WhatsApp message, or callback should be sent
or scheduled.

Only extract the customer's intent and explicit callback request.

========================
UPDATE RULES
============

1. Use both conversation history and the current transcript.

2. The current transcript has priority because it is the newest customer
   information.

3. Preserve valid information from previous turns.

4. If the customer gives newer contradictory information, use the newest
   customer-provided information.

5. Do not treat assistant suggestions or questions as customer requirements.

6. Do not invent missing information.

7. Do not infer budget, product count, features, timeline, or intent without
   sufficient evidence.

8. Keep intent stable unless new customer evidence justifies changing it.

9. Callback request and lead intent must always be evaluated independently.

10. If a callback is requested but date/time is missing:
    requested = true
    preserve the known callback information
    set only missing fields to null.

11. If no callback is requested:
    requested = false
    date = null
    time = null
    timezone = null.

12. Return only the requested structured output.

========================
OUTPUT
======

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





