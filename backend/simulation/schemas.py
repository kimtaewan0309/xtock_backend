from pydantic import BaseModel, Field
from typing import Optional, Literal


class CreateAccountRequest(BaseModel):
    nickname: str
    email: Optional[str] = None


class BuySellRequest(BaseModel):
    user_id: str
    symbol: str
    quantity: int = Field(gt=0)


class AutoTradeRequest(BaseModel):
    user_id: str
    symbol: str
    type: Literal["BUY", "SELL"]
    target_price: float = Field(gt=0)
    quantity: int = Field(gt=0)