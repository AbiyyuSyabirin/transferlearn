import os
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
import secrets
from io import BytesIO
from PIL import Image
from typing import Tuple
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from pydantic import BaseModel

app = FastAPI()  # Create the FastAPI app
device = torch.device('cpu')

API_KEY = "atmin1234"  # Ganti dengan API key Anda
api_key_header = APIKeyHeader(name="access_token", auto_error=True)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials")

# Preprocessing
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet50(pretrained=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.load_state_dict(torch.load('models/best_model_params.pt', map_location=torch.device('cpu')))
model.eval()

device = torch.device('cpu')
model.to(device)

class_names = ['goats', 'sheeps']

def read_file(data) -> Tuple[Image.Image, Tuple[int, int]]:
    img = Image.open(BytesIO(data)).convert('RGB')
    img_resized = img.resize((256, 256), resample=Image.BICUBIC)
    return img_resized, img_resized.size

@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    try:
        img, _ = read_file(await file.read())

        image = data_transforms(img)
        img_batch = image.unsqueeze(0)

        img_batch = img_batch.to(device)

        with torch.no_grad():
            outputs = model(img_batch)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            confidences = torch.softmax(outputs, dim=1)[0]

            confidence_dict = {
                class_names[i]: float(confidences[i])
                for i in range(len(class_names))
            }

        return {
            'class': predicted_class,
            'confidence': confidence_dict
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
