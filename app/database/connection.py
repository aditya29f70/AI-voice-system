from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession
)
import os
from dotenv import load_dotenv

load_dotenv()

db_url= (
    "postgresql+asyncpg://postgres:password@localhost:5442/voice_assistant"
)

engine= create_async_engine(
    db_url,
    echo =True
)

AsyncSessionLocal= async_sessionmaker(
    engine,
    class_= AsyncSession,
    expire_on_commit=False
)