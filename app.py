import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from main import TicketClassificationSystem

app = FastAPI(title="AI Customer Support Ticket Classifier API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Classifier
classifier_system = TicketClassificationSystem()

@app.on_event("startup")
def load_model():
    try:
        classifier_system.load_model()
    except Exception as e:
        print("Model not loaded. Try running `python main.py --train` first.")

class TicketRequest(BaseModel):
    ticket_text: str

@app.post("/api/predict")
def predict_ticket(request: TicketRequest):
    if not request.ticket_text.strip():
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty.")
    
    if not classifier_system.is_trained:
        raise HTTPException(status_code=500, detail="Model is not trained yet.")
        
    try:
        result = classifier_system.predict(request.ticket_text)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
