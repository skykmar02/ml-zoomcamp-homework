
import pickle

from typing import Dict, Any

from fastapi import FastAPI
import uvicorn

app = FastAPI(title= 'churn_prediction')



with open('model.bin', 'rb') as f_in:
  pipeline = pickle.load(f_in)


def predict_single(customer):
   result = pipeline.predict_proba(customer)[0, 1]
   return float(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
   prob = predict_single(customer)

   return {
      "churn_probability": prob,
      "churn": bool(prob>=0.5)
   }

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=True)
