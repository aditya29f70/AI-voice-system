from enum import Enum

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    String
)

from sqlalchemy.orm import Mapped, mapped_column
from app.database.database import Base

class CallbackStatus(str,Enum):
    SCHEDULED = "scheduled"
    CALLING = "calling"
    COMPLETED = "completed"
    FAILED = "failed"


class Customer(Base):
    __tablename__='customers'

    id: Mapped[int]= mapped_column(primary_key=True)

    phone_number: Mapped[str]= mapped_column(
        String(20),
        unique=True,
        index=True
    )

class Callback(Base):
    __tablename__= "callbacks"

    id: Mapped[int]= mapped_column(primary_key=True)

    customer_id: Mapped[int]= mapped_column(
        ForeignKey('customers.id')
    )
    scheduled_at: Mapped[datetime]= mapped_column(
        DateTime(timezone=True),
        index=True
    )

    timezone: Mapped[str]= mapped_column(
        String,
        default="Asia/Kolkata"
    )

    status: Mapped[CallbackStatus]= mapped_column(
        SQLEnum(CallbackStatus),
        default=CallbackStatus.SCHEDULED,
        index=True
    )
    previous_thread_id: Mapped[str: None]= mapped_column(
        String,
        nullable=True
    )
    error_message: Mapped[str: None]= mapped_column(
        String,
        nullable=True
    )
