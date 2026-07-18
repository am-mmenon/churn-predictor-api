from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the FastAPI Application
app = FastAPI(title="D2C Customer Churn Predictor API")

# 2. Load the trained machine learning model at startup
try:
    model = joblib.load("churn_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model file: {e}")

# 3. Define what incoming customer data MUST look like (Data Validation)
class CustomerData(BaseModel):
    recency: int
    frequency: int
    monetary: float

# 4. Create a basic Home Route to check if the server is alive
@app.get("/")
def home():
    return {
        "status": "API is online", 
        "model_loaded": model is not None
    }

# 5. Create the Prediction Route (Where data goes to get analyzed)
@app.post("/predict")
def predict_churn(customer: CustomerData):
    if not model:
        return {"error": "Machine learning model is not available."}
    
    # Convert the incoming data structure into a format the model understands
    input_df = pd.DataFrame([{
        'recency': customer.recency,
        'frequency': customer.frequency,
        'monetary': customer.monetary
    }])
    
    # Get the raw prediction (0 or 1) and the probability score (0.0 to 1.0)
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    
    # Categorize the risk level to make it easy for humans to read
    if probability > 0.7:
        risk = "High"
    elif probability > 0.4:
        risk = "Medium"
    else:
        risk = "Low"
        
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk
    }