from pydantic import BaseModel, Field
from typing import Optional, Literal


class CreateAccountRequest(BaseModel):
    nickname: str
    email: Optional[str] = None
    pin: str = Field(min_length=4, max_length=6)


class BuySellRequest(BaseModel):
    user_id: str
    symbol: str
    quantity: int = Field(gt=0)
    pin: str = Field(min_length=4, max_length=6)


class AutoTradeRequest(BaseModel):
    user_id: str
    symbol: str
    type: Literal["BUY", "SELL"]
    target_price: float = Field(gt=0)
    quantity: int = Field(gt=0)