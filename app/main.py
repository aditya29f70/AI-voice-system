from app.graph.graph import graph1
import sounddevice as sd
from scipy.io.wavfile import write
import asyncio


CONFIG= {"configurable":{"thread_id":"1"}}


async def main():


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


        init_state={
            "current_audio":audio_path
        }



        result= await graph1.ainvoke(init_state, config=CONFIG)

        print(f"Current User Transcript: {result['current_transcript']}\n")
        print(f"Current Ai Response: {result['current_response']}\n\n")
        

        if not result['should_continue']:
            break


if __name__=="__main__":
    asyncio.run(main())
