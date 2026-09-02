import os
from dotenv import load_dotenv
from app.graph.graph import graph1, graph2
import sounddevice as sd
from scipy.io.wavfile import write
import asyncio

load_dotenv()

call_id= os.getenv("MY_NUMBER")

GRAPH1_CONFIG= {"configurable":{"thread_id":f"{call_id}:fast"}}
GRAPH2_CONFIG= {"configurable":{"thread_id":f"{call_id}:lead"}}


async def main():

    find_lead_flag=True
    callback_message=""
    lead_update={"actions":{"whatsapp_sent_mid_call":False, "callback_scheduled":False}, "callback":{"callback_situation":None}}
    while True:
        sample_rate= 16000
        duration= 20

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



        result= await graph1.ainvoke(graph1_init_state, config=GRAPH1_CONFIG)

        print(f"Current User Transcript: {result['current_transcript']}\n")
        print(f"Current Ai Response: {result['current_response']}\n\n")
        print(result)
        

        if not result['should_continue']:
            break

        if result and find_lead_flag:
            graph2_init_state={
                "messages": result['messages'],
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





            


if __name__=="__main__":
    asyncio.run(main())
