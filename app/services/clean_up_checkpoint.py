from sqlalchemy import text
from app.database.connection import AsyncSessionLocal
import asyncio

async def clean_checkpointer():
    async with AsyncSessionLocal() as sesssion:
        await sesssion.execute(
            text("TRUNCATE TABLE checkpoints CASCADE")
        )

        await sesssion.commit()



# async def check_tables():
#     async with AsyncSessionLocal() as session:

#         result = await session.execute(
#             text("""
#                 SELECT table_name
#                 FROM information_schema.tables
#                 WHERE table_schema = 'public'
#             """)
#         )

#         rows = result.fetchall()

#         for row in rows:
#             print(row)

#         # await sesssion.commit()

if __name__=="__main__":
    asyncio.run(clean_checkpointer())