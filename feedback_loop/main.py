from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from pymongo import MongoClient
from typing import Optional
import os

# Connect to MongoDB
DATABASE_NAME = os.getenv("DATABASE_NAME")
MONGO_URI = os.getenv("MONGO_URI")
PRED_COLLECTION_NAME = os.getenv("PRED_COLLECTION_NAME")
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
prediction_collection = db[PRED_COLLECTION_NAME]

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

# OAuth2 scheme for Bearer token in Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Helper function to decode JWT token
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token"
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid token"
        )

app = FastAPI()

class Feedback(BaseModel):
    user_feedback: str  # e.g., "correct", "incorrect"
    correct_label_idx: Optional[int] = None  # Optional correct label idx, if incorrect

# GET route to fetch prediction info for a given dataset_name and image_id
@app.get("/feedback/{dataset_name}/{image_id}")
async def get_prediction_info(
    dataset_name: str,
    image_id: int,
    token: str = Depends(verify_token)
):
    # Find the prediction in the MongoDB collection
    prediction = prediction_collection.find_one({
        "dataset_name": dataset_name,
        "image_id": image_id
    })

    if not prediction:
        # If the prediction is not found, raise a 404 error
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Return the prediction details and feedback (if any)
    return {
        "image_id": prediction.get("image_id"),
        "dataset_name": prediction.get("dataset_name"),
        "predicted_class_idx": prediction.get("predicted_class_idx"),
        "predicted_class_label": prediction.get("predicted_class_label"),
        "user_feedback": prediction.get("user_feedback", "No feedback yet")
    }

# POST route to submit user feedback for a given dataset_name and image_id
@app.post("/feedback/{dataset_name}/{image_id}")
async def submit_feedback(
    dataset_name: str,
    image_id: int,
    feedback: Feedback,
    token: str = Depends(verify_token)
):
    # Find the prediction in the MongoDB collection
    prediction = prediction_collection.find_one({
        "dataset_name": dataset_name,
        "image_id": image_id
    })

    if not prediction:
        # If the prediction is not found, raise a 404 error
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Update the MongoDB document with the user's feedback
    update_result = prediction_collection.update_one(
        {"dataset_name": dataset_name, "image_id": image_id},
        {"$set": {"user_feedback": feedback.user_feedback, "correct_label": feedback.correct_label_idx}}
    )

    # Check if the document was modified
    if update_result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Failed to submit feedback")

    return {"message": "Feedback submitted successfully"}
