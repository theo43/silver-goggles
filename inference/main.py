import torch
from fastapi import FastAPI, Query, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from transformers import AutoModelForImageClassification, AutoProcessor
from pymongo import MongoClient
from gridfs import GridFS
from PIL import Image
from jose import JWTError, jwt
import io
import os
import ast

# FastAPI app initialization
app = FastAPI()

# Load ImageNet labels from the JSON file - JSON format not respected so create python dict directly
with open("/app/imagenet_labels.json", "r") as f:
    file_content = f.read()
imagenet_labels = ast.literal_eval(file_content)

# MongoDB connection parameters from environment variables
DATABASE_NAME = os.getenv("DATABASE_NAME")
MONGO_URI = os.getenv("MONGO_URI")

# Initialize MongoDB client and GridFS
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
fs = GridFS(db)

# Load Hugging Face model and processor for resnet-50
model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

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

# Ensure model is in eval mode
model.eval()

# Helper function to load image from GridFS and return a PIL image
def load_image_from_gridfs(file_id):
    # Fetch the image file from GridFS using file_id
    grid_out = fs.get(file_id)
    
    # Convert the image binary data into a PIL image
    image = Image.open(io.BytesIO(grid_out.read()))
    return image

# Helper function to preprocess the image for the model
def preprocess_image(image: Image):
    # Resize the image to the expected input size for ResNet-50 (224x224)
    image = image.resize((224, 224))

    # Preprocess the image using the processor, with padding and batching
    inputs = processor(images=[image], return_tensors="pt", padding=True)  # Batch of 1
    return inputs["pixel_values"]

# Route for inference based on image_id from MongoDB
@app.get("/predict/{collection_name}/{image_id}")
def predict(
    collection_name: str,
    image_id: int,
    token: str = Depends(verify_token)
):
    # Fetch the metadata (file_id) for the image from MongoDB using the image_id
    collection = db[collection_name]
    record = collection.find_one({"image_id": image_id})

    if not record:
        raise HTTPException(status_code=404, detail="Image not found")

    # Extract the file_id from the record to load the image from GridFS
    file_id = record.get("file_id")
    if not file_id:
        raise HTTPException(status_code=404, detail="File ID not found in record")

    # Load the image from GridFS using the file_id
    image = load_image_from_gridfs(file_id)

    # Preprocess the image for the model
    image_tensor = preprocess_image(image)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = torch.argmax(predictions, dim=-1).item()

    # Return the predicted class index
    return {
        "image_id": image_id,
        "predicted_class_idx": predicted_class_idx,
        "predicted_class_label": imagenet_labels[predicted_class_idx]
    }
