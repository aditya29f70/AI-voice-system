import os
import sys
from dotenv import load_dotenv
from app.graph.graph import builder1, builder2

from app.audio.adaptive_recorder import AdaptiveRecorder
from app.services.stt_service import STTService

from langchain_core.messages import HumanMessage

import asyncio
from sqlalchemy import select

from app.database.connection import AsyncSessionLocal
from app.database.models import Customer

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.graph.nodes.after_call_end import after_call_email

load_dotenv()

call_id= os.getenv("MY_NUMBER")

DB_URL= os.getenv("DATABASE_URL")

async def main(call_id):
    customer_phone= os.getenv("MY_NUMBER")

    async with AsyncSessionLocal() as session:
        result= await session.execute(
            select(Customer).where(
                Customer.phone_number== customer_phone
            )
        )
        customer= result.scalar_one_or_none()
        if customer is None:
            customer= Customer(
                phone_number= customer_phone
            )

            session.add(customer)
            await session.commit()
    
    GRAPH1_CONFIG= {"configurable":{"thread_id":f"{customer.id}:{call_id}:fast"}}
    GRAPH2_CONFIG= {"configurable":{"thread_id":f"{customer.id}:{call_id}:lead"}}

    find_lead_flag=True
    callback_message=""

    callback_requested= False
    email_sent_mid_call=False
    callback_scheduled= False

    lead_update={"actions":{"whatsapp_sent_mid_call":False, "callback_scheduled":False}, "callback":{"callback_situation":None}}

    recorder = AdaptiveRecorder()
    stt = STTService()
    print("\nVoice assistant started.")

    result= {'should_continue':True}
    while result['should_continue']:
        try:

            # --------------------------------
            # Wait for user speech
            # --------------------------------

            audio = recorder.record_utterance()

            if audio is None:
                continue

            # --------------------------------
            # STT
            # --------------------------------

            print("\nTranscribing...")

            text = stt.transcribe(audio)

            print(
                f"\nUser: {text}"
            )

            if not text:
                continue

            # --------------------------------
            # YOUR LANGGRAPH
            # --------------------------------

            # result = graph.invoke(...)
            #
            # response = result[...]

            graph1_init_state={
                "messages":[HumanMessage(content=text)],
                "current_transcript":text,
                "callback_situation": callback_message if callback_message else None,
                "callback_requested":callback_requested,
                "email_sent_mid_call":email_sent_mid_call,
                "callback_scheduled": callback_scheduled
            }



            async with AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer:
                await checkpointer.setup()

                graph1= builder1.compile(checkpointer=checkpointer)
                graph2= builder2.compile(checkpointer=checkpointer)

                result= await graph1.ainvoke(graph1_init_state, config=GRAPH1_CONFIG)

                print(f"Current Ai Response: {result['current_response']}\n\n")
                print(result)
                


                if result and find_lead_flag:
                    graph2_init_state={
                        "messages": result['messages'],
                        "thread_id": f"{customer.id}:{call_id}:lead",
                        "customer_id":customer.id if customer else None,
                        "customer_phone": os.getenv("MY_NUMBER"),
                        "customer_email": "adityakumar81raj@gmail.com",
                        "current_transcript": result['current_transcript'],
                        "current_response": result['current_response'],
                        "actions":{"whatsapp_sent_mid_call":lead_update['actions']['whatsapp_sent_mid_call'], "callback_scheduled": lead_update['actions']['callback_scheduled']},
                        "callback":{"callback_situation": lead_update['callback']['callback_situation']}
                    }

                    lead_update= await graph2.ainvoke(graph2_init_state, config=GRAPH2_CONFIG)

                    print(f"\nLead updatelead_update{lead_update}\n\n")

                    if lead_update['callback']['callback_situation']:
                        callback_message= lead_update['callback']['callback_situation']

                    callback_requested=lead_update['callback']['requested']
                    email_sent_mid_call=lead_update['actions']['whatsapp_sent_mid_call']
                    callback_scheduled =lead_update['actions']['callback_scheduled']

                    if lead_update['actions']['whatsapp_sent_mid_call'] and lead_update['actions']['callback_scheduled']:
                        find_lead_flag=False

        except KeyboardInterrupt:

            print(
                "\nStopping assistant."
            )

            break

        except Exception as e:

            print(
                f"\nError: {e}"
            )

    final_result= await after_call_email(lead_update)

    print(final_result)





if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
            


if __name__=="__main__":
    asyncio.run(main(call_id))
