from app.services.email_service import EmailService
from app.services.llm import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


email_service = EmailService()


EMAIL_PROMPT=ChatPromptTemplate.from_template("""
You are writing a follow-up email to a customer on a phone call or min of a call.

Write a personalized, professional, and friendly email.

IMPORTANT:
- Use specific information from the conversation.
- Do NOT write a generic summary.
- Mention only information actually discussed.
- Include what the customer said they want.
- Include the budget if mentioned.
- Include the timeline if mentioned.
- Include specific requested features if mentioned.
- Do not invent missing information.
- Do not mention that the customer was classified as "hot".
- Keep the email concise and natural.

My name is -> Aditya Kumar (sender)

Conversation history:
{conversation_history}

Structured information extracted from the conversation:

Business / product:
{what_they_sell}

Budget:
{budget}

Number of products:
{number_of_products}

Timeline:
{timeline}

Features:
{features}

Write the complete email now.
""")


async def hot_action(state):
    lead= state['lead']

    conversation_history= "\n".join(f"- {message.type}: {message.content}" for message in state['messages'])

    message= (
        "Hi!"
        "Thank you for your interest."
        "We've noted your requirements and our team will assist you further."
    )

    parser= StrOutputParser()

    chain= EMAIL_PROMPT|llm|parser

    response= await chain.ainvoke({
        "conversation_history":conversation_history,

        "what_they_sell": lead.get("what_they_sell"),
        "budget": lead.get("budget"),
        "number_of_products": lead.get("number_of_products"),
        "timeline": lead.get("timeline"),
        "features": lead.get("features")
    })


    result= await email_service.send_email(
        to_email= state["customer_email"],
        subject="Following up on our conversation",
        body=response
    )

    return {"actions":{"whatsapp_sent_mid_call":result["success"], "callback_scheduled": state['actions']['callback_scheduled']}, "lead":state['lead'], "callback": state['callback']}

   