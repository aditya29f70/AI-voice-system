from app.database.connection import engine
from app.database.database import Base
import asyncio

# Important: import models
from app.database import models

async def init_db():
    async with engine.begin() as conn:

        await conn.run_sync(
            Base.metadata.create_all
        )


asyncio.run(init_db())