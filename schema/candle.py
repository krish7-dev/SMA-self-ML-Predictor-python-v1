from pydantic import BaseModel
from datetime import datetime
from typing import Optional
class Candle(BaseModel):
    timestamp: Optional[datetime] = None
    open: float
    high: float
    low: float
    close: float
    volume: float
