import os
import io
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer
from datasets import load_dataset
from pymongo import MongoClient
from gridfs import GridFS
from PIL import Image, ImageDraw, ImageFont
from jose import JWTError, jwt
from fastapi.responses import StreamingResponse

# FastAPI app initialization
app = FastAPI()

# Logger setup
logger = logging.getLogger("uvicorn")

# MongoDB connection parameters from environment variables
DATABASE_NAME = os.getenv("DATABASE_NAME")
MONGO_URI = os.getenv("MONGO_URI")

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

# Initialize MongoDB client and GridFS
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
fs = GridFS(db)

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

# Route to fetch and serve the dataset, with token verification
@app.post("/collect_data/{hugging_face_repo}/{dataset_name}")
def collect_data(
    hugging_face_repo: str,
    dataset_name: str,
    token: str = Depends(verify_token)
):
    # Load the dataset from Hugging Face
    try:
        logger.info(f"Loading dataset {dataset_name} from {hugging_face_repo}")
        dataset = load_dataset(
            f"{hugging_face_repo}/{dataset_name}",
            cache_dir="/tmp/huggingface_datasets"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error loading dataset {hugging_face_repo}/{dataset_name}: {str(e)}"
        )
    
    # Iterate through the dataset and insert into MongoDB GridFS
    logger.info(f"Ingesting {len(dataset['train'])} records into MongoDB collection '{dataset_name}'...")
    for i, record in enumerate(dataset['train']):
        # Convert the PIL image to bytes for storing in GridFS
        image = record['image']
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)  # Reset the BytesIO stream position

        # Insert image into GridFS and store the file ID
        file_id = fs.put(image_bytes, filename=f"image_{i+1}.jpg")
        
        # Create the record with image metadata
        record_json = {
            "image_id": i + 1,
            "file_id": file_id,
            "label": record['labels']
        }

        db[dataset_name].insert_one(record_json)

    # Return a response with the number of records ingested
    return {
        "message": f"HuggingFace dataset {hugging_face_repo}/{dataset_name} ingested successfully in {dataset_name} collection",
        "count": len(dataset['train'])
    }

# Route to fetch an image from MongoDB GridFS by image_id, modify it, and return it for visualization
@app.get("/get_image/{collection_name}/{image_id}")
def get_image(
    collection_name: str,
    image_id: int,
    token: str = Depends(verify_token)
):
    logger.info(f"Fetching image {image_id} from collection {collection_name}")
    collection = db[collection_name]
    record = collection.find_one({"image_id": image_id})

    if record:
        file_id = record.get("file_id")
        label = record.get("label")
        
        # Fetch the image from GridFS
        image_file = fs.get(file_id)
        image = Image.open(io.BytesIO(image_file.read()))

        # Modify the image to write the text "image_id: {image_id} - label: {label}" on the top left corner
        draw = ImageDraw.Draw(image)
        
        # Try loading a font, or use a default font if none is found
        try:
            font = ImageFont.truetype("arial.ttf", size=20)
        except IOError:
            font = ImageFont.load_default()

        text = f"image_id: {image_id} - label: {label}"
        draw.text((10, 10), text, font=font, fill="white")  # Write text on the image
        
        # Save the modified image to a buffer
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        # Return the image as a StreamingResponse
        return StreamingResponse(buf, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Image not found")


@app.delete("/clean/{collection_name}")
async def clean_collections(
    collection_name: str,
    token: str = Depends(verify_token)
):
    try:
        logger.info(f"Cleaning collection {collection_name}...")
        collection = db[collection_name]
        collection.delete_many({})  # Remove all records from the collection
        fs.delete_many({})  # Remove all files from GridFS
        return {"message": f"Collections {collection_name} cleaned successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clean collections {collection_name}: {e}"
        )
