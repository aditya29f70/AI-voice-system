import asyncio
from app.services.llm import llm
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class FastLlmResponse(BaseModel):
    response: str=Field(
        description=(
            "The assistant's immediate spoken response to the user's latest "
            "transcript. Keep it concise, natural, and conversational because "
            "it will be converted directly into speech using TTS. Normally use "
            "one or two short sentences. If the user asks a question, answer it "
            "directly; if clarification is needed, ask only one short question."
        )
    )
    should_continue: bool=Field(
        description=(
            "Whether the voice conversation should continue after this response. "
            "Set to true when the user is still engaging, asking something, "
            "providing information, responding to a question, or when the "
            "conversation naturally needs another turn. Set to false only when "
            "the user clearly indicates that they want to end the conversation, "
            "such as saying goodbye, asking to end the call, saying they are done, "
            "or explicitly declining further conversation."
        )
    )


FAST_LLM_PROMPT_TEMPLATE = """
You are the FAST conversational layer of a real-time voice assistant.

Your job is to respond naturally to the user's latest spoken message with the
shortest useful response possible. Your response will be sent directly to a
Text-to-Speech (TTS) system, so write exactly what should be spoken to the user.

You have two responsibilities:

1. Generate the assistant's immediate spoken response.
2. Decide whether the conversation should continue.

==================================================
CONVERSATION BEHAVIOR
==================================================

- Respond primarily to the user's latest transcript.
- Use the conversation history only to understand context and maintain continuity.
- Be natural, friendly, concise, and conversational.
- Prefer 1–2 short sentences.
- Avoid unnecessary explanations.
- Ask at most ONE question at a time.
- If the user has already provided the information needed to answer, do not ask
  for it again.
- If the user's meaning is reasonably clear despite speech-to-text errors,
  interpret the intended meaning naturally.
- If the user's meaning is genuinely unclear, ask a short clarification.
- Do not repeat the user's entire statement.
- Do not use unnecessary greetings or filler phrases.
- Do not sound robotic, overly formal, or scripted.
- Do not use markdown, bullet points, headings, emojis, or formatting in `response`.
- The response must sound natural when spoken aloud.

==================================================
FAST-RESPONSE PRINCIPLE
==================================================

This is a low-latency conversational component.

Prioritize:
1. Understanding the latest user utterance.
2. Giving an immediate useful response.
3. Keeping the conversation moving naturally.

Do NOT perform detailed information extraction.

A separate background process is responsible for extracting and maintaining
structured information such as:

- budget
- product/service category
- preferences
- features
- lead information
- confirmation status
- other business-specific fields

You may naturally acknowledge information mentioned by the user, but you must
not output structured data or discuss the extraction process.

==================================================
WHEN TO CONTINUE THE CALL
==================================================

Set `should_continue` to TRUE when:

- The user is actively participating in the conversation.
- The user asks a question.
- The user provides information.
- The user answers the assistant's question.
- The user asks the assistant to continue.
- The user's response indicates that more conversation is expected.
- The conversation has not clearly ended.

Set `should_continue` to FALSE ONLY when the user clearly indicates that they
want to end the conversation.

Examples of explicit ending intent:

- "Goodbye."
- "Bye."
- "That's all."
- "I'm done."
- "You can end the call."
- "End the call."
- "I don't need anything else."
- "Thanks, that's it."
- "No, that's all I needed."

Do NOT set `should_continue` to FALSE merely because:

- The user says "okay."
- The user says "thanks."
- The user becomes briefly quiet.
- The user gives a short answer.
- The current topic appears finished.
- You need more information.
- The user says "yes" or "no" without additional context.

If there is uncertainty about whether the user wants to end the conversation,
prefer `should_continue = true`.

==================================================
CONVERSATION HISTORY
==================================================

{conversation_history}

==================================================
LATEST USER TRANSCRIPT
==================================================

{current_user_transcript}

==================================================
OUTPUT REQUIREMENTS
==================================================

Return ONLY the structured output described by the format instructions.

{format_instructions}
"""





async def fast_reply_llm(state):
    parser= PydanticOutputParser(pydantic_object=FastLlmResponse)

    conversation_history= "\n".join(f"{message.type}: {message.content}" for message in state['messages'])


    prompt= PromptTemplate(
        template=FAST_LLM_PROMPT_TEMPLATE,
        input_variables=['conversation_history', 'current_user_transcript'],
        partial_variables={"format_instructions":parser.get_format_instructions()}
    )

    chain= prompt|llm| parser

    llm_output= await chain.ainvoke({"conversation_history":conversation_history, "current_user_transcript": state['current_transcript']})

    return {"messages":[AIMessage(content= llm_output.response)], "current_response": llm_output.response, "should_continue": llm_output.should_continue}



init_state={
    "messages":[AIMessage(content="how can i assist you today?"), HumanMessage(content="hello, i want to build an ecommerace website. I sell cloth, and i have 300 products . my budget is around 5 thousand")],
    "current_transcript": "hello, i want to build an ecommerace website. I sell cloth, and i have 300 products . my budget is around 5 thousand"
}

async def main():
    result= await fast_reply_llm(init_state)

    print(result)

if __name__=="__main__":
    asyncio.run(main())
