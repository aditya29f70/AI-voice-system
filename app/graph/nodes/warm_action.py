from datetime import datetime
from datetime import date
from datetime import time
from zoneinfo import ZoneInfo
from app.database.models import Callback, CallbackStatus
from app.database.connection import AsyncSessionLocal
import asyncio


async def warm_action(state):

    callback_situation=""
    if not state['callback']['date']:
        callback_situation+= "date of callback is not provided,"

    if not state['callback']['time']:
        callback_situation+= "time to callback is not provided"

    if callback_situation:
        # return {"callback": {"callback_situation": callback_situation}}
        return {
            "callback": {
                "callback_situation": callback_situation,
                "requested": state["callback"]["requested"],
                "date": state["callback"]["date"],
                "time": state["callback"]["time"],
                "timezone": state["callback"]["timezone"],
            },
            "actions": {
                "callback_scheduled": False,
                "whatsapp_sent_mid_call": state["actions"]["whatsapp_sent_mid_call"],
            },
            "lead": state["lead"],
        }
    
    scheduled_at= datetime.combine(
        state['callback']['date'],
        state['callback']['time']
    )

    timezone = (
        state['callback'].get("timezone")
        or "Asia/Kolkata"
    )


    scheduled_at= scheduled_at.replace(
        tzinfo=ZoneInfo(timezone)
    )

    async with AsyncSessionLocal() as session:
        try:
            new_callback = Callback(
                customer_id=state["customer_id"],
                scheduled_at=scheduled_at,
                timezone=timezone,
                status=CallbackStatus.SCHEDULED,
                previous_thread_id=state["thread_id"],
            )

            session.add(new_callback)
            await session.commit()

            await session.refresh(new_callback)

            print("Callback successfully saved!")
            print("Callback ID:", new_callback.id)

        except Exception as e:
            await session.rollback()
            print("DATABASE ERROR:", repr(e))
            raise

    

    # For now, simulate scheduling
    print("Scheduling callback...")
    print(f"Date: {state['callback']['date']}")
    print(f"Time: {state['callback']['time']}")
    print(f"Timezone: {state['callback']['timezone']}")

    # return {"actions":{"callback_scheduled":True}}
    return {"actions":{"callback_scheduled":True, "whatsapp_sent_mid_call": False}, "lead":state['lead'], "callback": state['callback']}


# init_state={
#     "callback":{
#         "date": date(2026, 9, 5),
#         "time": time(16, 0),
#         "timezone": None
#     },
#     "customer_id":4,
#     "thread_id":"1"
# }
# async def call_this_fun():
#     result= await warm_action(init_state)
#     print(result)
#     return result

# if __name__=='__main__':
#     asyncio.run(call_this_fun())