import os
import sys
from dotenv import load_dotenv
from app.graph.graph import builder1, builder2
import sounddevice as sd
from scipy.io.wavfile import write
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
    lead_update={"actions":{"whatsapp_sent_mid_call":False, "callback_scheduled":False}, "callback":{"callback_situation":None}}
    while True:
        sample_rate= 16000
        duration= 13

        print("speak now.. ")


        audio= sd.rec(
            int(duration* sample_rate),
            samplerate = sample_rate,
            channels=1,
            dtype="int16"

        )

        sd.wait()

        write("audio.wav", sample_rate, audio)
        print("Saved as audio.wav")

        audio_path= "audio.wav"


        graph1_init_state={
            "current_audio":audio_path,
            "callback_situation": callback_message if callback_message else None
        }

        async with AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer:
            await checkpointer.setup()

            graph1= builder1.compile(checkpointer=checkpointer)
            graph2= builder2.compile(checkpointer=checkpointer)

            result= await graph1.ainvoke(graph1_init_state, config=GRAPH1_CONFIG)

            print(f"Current User Transcript: {result['current_transcript']}\n")
            print(f"Current Ai Response: {result['current_response']}\n\n")
            print(result)
            

            if not result['should_continue']:
                break

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

                if lead_update['actions']['whatsapp_sent_mid_call'] and lead_update['actions']['callback_scheduled']:
                    find_lead_flag=False

    final_result= await after_call_email(lead_update)

    print(final_result)
    




if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
            


if __name__=="__main__":
    asyncio.run(main(call_id))
