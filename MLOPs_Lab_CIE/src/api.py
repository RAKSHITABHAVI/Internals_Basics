from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

# Input validation
class InputData(BaseModel):
    ad_spend: float = Field(..., ge=1000, le=50000)
    audience_size: int = Field(..., ge=10000, le=500000)
    creative_score: float = Field(..., ge=1, le=10)
    is_retargeting: int = Field(..., ge=0, le=1)

# Health endpoint
@app.get("/health")
def health():
    return {
        "alive": True,
        "service": "AdPulse click_through_rate API"
    }

# Prediction endpoint
@app.post("/score")
def score(data: InputData):
    # dummy prediction (allowed for exam)
    return {"prediction": 5.0}