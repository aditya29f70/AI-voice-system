import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from app.services.email_service import EmailService

from app.services.llm import llm

load_dotenv()

class FinalEmail(BaseModel):
    subject: str 
    body: str

AFTER_CALL_EMAIL_PROMPT = """
You are writing a personalized follow-up email after a real phone
conversation between a customer and a website/application development business.

my name - Aditya Kumar (sender)
Write a natural, professional email based ONLY on the information provided.

==================================================
FULL CONVERSATION
==================================================

{conversation_history}

==================================================
CUSTOMER INFORMATION
==================================================

Customer phone number:
{customer_phone}

==================================================
LEAD INFORMATION
==================================================

Business / project:
{what_they_sell}

Budget:
{budget}

Number of products:
{number_of_products}

Timeline:
{timeline}

Requested features:
{features}

Lead intent:
{intent}

==================================================
CALLBACK INFORMATION
==================================================

Callback requested:
{callback_requested}

Callback date:
{callback_date}

Callback time:
{callback_time}

Timezone:
{callback_timezone}

Reason / situation:
{callback_situation}

Callback successfully scheduled:
{callback_scheduled}

==================================================
PREVIOUS ACTIONS
==================================================

WhatsApp sent during the call:
{whatsapp_sent_mid_call}

Email already sent during the call:
{email_sent_mid_call}

==================================================
YOUR TASK
==================================================

Write a personalized follow-up email that reflects the actual conversation.

The email must:

1. Include the SPECIFIC context of what the customer discussed.

2. Mention relevant details such as:
   - what they want to build
   - their business or project
   - budget, if mentioned
   - timeline, if mentioned
   - requested features, if mentioned

3. Sound like a real person writing after a phone conversation.
   Do not make it sound like a database record or AI-generated report.

4. Use callback information accurately.

   - If callback_scheduled is true, naturally mention the confirmed callback
     date and time.

   - If callback_requested is true but callback_scheduled is false,
     do not claim that the callback is confirmed.

   - If no callback was requested, do not mention scheduling unnecessarily.

5. Consider previous actions.

   - Do not unnecessarily repeat or duplicate a message that was already
     sent during the call.

6. Do not invent any information.

7. If budget, timeline, features, or other information was not discussed,
   simply omit it naturally instead of saying "not available".

8. Keep the tone friendly, professional, and personal.

Return ONLY the email body.

{format_instructions}
"""

async def after_call_email(state):
    conversation_history= "\n".join(f"- {message.type}: {message.content}" for message in state['messages'])

    lead= state.get('lead', {})
    callback= state.get('callback', {})
    actions = state.get('callback', {})

    parser= PydanticOutputParser(pydantic_object=FinalEmail)

    prompt= PromptTemplate(
        template=AFTER_CALL_EMAIL_PROMPT,
        input_variables=[
            "conversation_history",
            "customer_phone",
            "what_they_sell",
            "budget",
            "number_of_products",
            "timeline",
            "features",
            "intent",
            "callback_requested",
            "callback_date",
            "callback_time",
            "callback_timezone",
            "callback_situation",
            "callback_scheduled",
            "whatsapp_sent_mid_call",
            "email_sent_mid_call"
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain= prompt|llm|parser

    result=await chain.ainvoke({
    "conversation_history": conversation_history,

    # Lead
    "what_they_sell": lead.get("what_they_sell"),
    "budget": lead.get("budget"),
    "number_of_products": lead.get("number_of_products"),
    "timeline": lead.get("timeline"),
    "features": ", ".join(lead.get("features", [])),
    "intent": lead.get("intent"),

    # Callback
    "callback_requested": callback.get("requested"),
    "callback_date": callback.get("date"),
    "callback_time": callback.get("time"),
    "callback_timezone": callback.get("timezone"),
    "callback_situation": callback.get("callback_situation"),

    # Actions
    "whatsapp_sent_mid_call": actions.get("whatsapp_sent_mid_call"),
    "callback_scheduled": actions.get("callback_scheduled"),
    "email_sent_mid_call": actions.get("email_sent_mid_call"),

    # Customer
    "customer_phone": state.get("customer_phone"),
    })

    email_service= EmailService()

    BASE_DIR = Path.cwd()

    resume_path = BASE_DIR / "app" / "assets" / "resume.pdf"
    arch_path = BASE_DIR / "app" / "assets" / "system_arch.png"

    await email_service.send_email(
        to_email=state["customer_email"],
        subject=result.subject,
        body=result.body,
        attachments=[str(resume_path), str(arch_path)]
    )



    return "\nhave send final email!!!!"