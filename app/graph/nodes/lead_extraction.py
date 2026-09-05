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

The lead intent represents the customer's CURRENT BUYING INTENT
and READINESS to pursue the project.

Intent is NOT a measure of:

* how much information the customer provided
* how friendly the customer was
* how long the conversation lasted
* whether the customer answered questions
* whether the customer has a possible use case
* whether a callback was requested
* whether an email was sent

Allowed values:

* hot
* warm
* cold
* null

Evaluate the customer's overall behavior across the conversation.

==================================================
HOT
===

Classify as HOT when the customer demonstrates clear current intent
to move toward getting the project built.

Strong HOT signals include:

* explicitly wants to proceed
* wants to start the project
* asks how to begin
* asks for next steps
* agrees to move forward
* asks for implementation
* requests a proposal or quotation AND appears serious about proceeding
* asks practical questions about getting the project built
* discusses implementation details with the intention of moving forward
* discusses budget, payment, timeline, or project execution seriously
* indicates a concrete or reasonably near-term plan to start
* clearly states that they want the business to build the project

Do NOT classify as HOT merely because the customer:

* describes a possible website
* answers discovery questions
* gives several requirements
* asks general questions
* is curious
* is browsing
* says they may do it someday

HOT means there is meaningful evidence of CURRENT buying intent.

==================================================
WARM
====

Classify as WARM when the customer has a genuine potential need
or meaningful interest, but is not currently ready or committed
to move forward.

Typical WARM signals include:

* has a real problem they want solved
* genuinely wants a website/application but is still evaluating
* is considering getting the project built
* wants to compare options before deciding
* is interested but needs to think about it
* expects to pursue the project later but has a meaningful intention
* discusses the project seriously even though the timing is uncertain
* asks relevant questions about pricing, features, implementation,
  or suitability while seriously considering the project

WARM requires more than simple curiosity.

A customer should generally NOT be WARM when they repeatedly describe
themselves as merely browsing or casually curious without a real plan
to pursue the project.

==================================================
COLD
====

Classify as COLD when the customer shows little or no CURRENT buying
intent, even if they remain polite, answer questions, or describe
a possible use case.

COLD does NOT require explicit rejection.


Important:

A customer can provide detailed project information and STILL be COLD
if their overall behavior shows that they are only browsing or casually
exploring.

For example:

Customer:
"I have an old website, but I'm not actively planning anything.
I'm just curious."

Later:
"Maybe sometime later, but no rush."

Later:
"I haven't thought about budget. Just browsing."

This should be classified as:

intent = cold

because the customer's behavior shows no meaningful current buying intent.

Do NOT classify this customer as WARM merely because they:

* have a business
* have an old website
* describe possible features
* describe what they would put on the website
* answer discovery questions
* say they might do something later

The existence of a possible future project is not sufficient for WARM.

==================================================
INTENT PRIORITY
==============

When signals conflict, prioritize explicit statements about the
customer's current intentions over the amount of project information
they provide.

For example:

Detailed requirements + "I'm just browsing" = COLD

Detailed requirements + "I want to build this, but I'm deciding
between providers" = WARM

Detailed requirements + "I want to get this started next month" = HOT

Possible future project + no current plan = COLD or WARM depending
on whether there is genuine intent to pursue it.

Casual curiosity alone = COLD.
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

WARM → COLD

Change WARM to COLD when new customer statements clearly show that
their previous interest was only exploratory or that they currently
have no meaningful intention to pursue the project.

Examples:

"I was just curious."

"I'm not actually planning to do anything."

"I'm only browsing."

"I don't have any plans right now."

"I was just checking what you offer."

These can justify WARM → COLD even without explicit rejection.

When intent cannot yet be determined:

intent = null
confirmed = false

Do not force a classification.

COLD → WARM

Allow COLD → WARM when the customer later demonstrates genuine interest
or a real potential need.

Example:

"I'm just browsing."

...

"Actually, our current website is causing us problems and I've been
thinking about replacing it."

This can justify COLD → WARM.


WARM → HOT

Allow when the customer moves from consideration to concrete intent.

Example:

"I've been considering it."

...

"Actually, I'd like to get this started next month. What would you need
from me?"

→ HOT


==================================================
CONVERSATION OBJECTIVE
======================

The goal is NOT to collect every lead field.

The goal is to understand whether the customer has a genuine current
need or interest and, when appropriate, learn enough about their project
to continue a useful conversation.

Prioritize the customer's experience over completing a checklist.

If the customer is highly interested:
    explore the project naturally.

If the customer is uncertain:
    understand what is causing their uncertainty.

If the customer is merely curious:
    answer their questions and optionally ask one lightweight question.

If the customer repeatedly indicates that they are only browsing or
have no current plans:
    stop intensive qualification and allow the conversation to remain
    low-pressure.

Never manufacture urgency.

Never repeatedly ask for information after the customer has indicated
that they are not actively planning a project.

========================
CONFIRMED
=========

confirmed represents confidence that the assigned intent accurately
reflects the customer's CURRENT buying intent.

Set confirmed = true when there is sufficient behavioral evidence
to distinguish the customer's intent.

Examples:

Clearly says "I'm just browsing" + repeatedly says no plans
→ cold, confirmed = true

Has a genuine project but says "I'm still deciding"
→ warm, confirmed = true

Clearly wants to start next month
→ hot, confirmed = true

Customer gives mixed signals and their actual intention is unclear
→ intent = null, confirmed = false

Do not require the customer to explicitly say:
"I'm a cold lead."

Infer intent from their actual statements and behavior.

However, do not infer positive buying intent merely from politeness,
cooperation, curiosity, or answering questions.

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





