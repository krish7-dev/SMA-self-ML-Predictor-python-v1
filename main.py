from fastapi import FastAPI

app = FastAPI()

MODEL = None
METRICS = {
    "accuracy": None,
    "feature_importance": None,
    "last_trained": None
}

@app.get("/")
def root():
    return {"message": "API is running ✅"}
