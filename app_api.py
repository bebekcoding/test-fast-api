from fastapi import FastAPI
from model.simple_model import MyModel
import torch
from typing import List
from utils import config

# load model
model = MyModel(
    input_size=config.INPUT_SIZE,
    output_size=config.OUTPUT_SIZE
)
model.load_state_dict(torch.load("checkpoints/checkpoints-1000.pth"))
model.eval()

myapp = FastAPI()

@myapp.get("/")
def index():
    return {"message": "API siap digunakan"}

@myapp.post("/predict")
def predict(features: list[float]):
    features = torch.tensor(features).float().unsqueeze(0)

    with torch.inference_mode():
        output = model(features)

        prediction = torch.argmax(output, dim=1).item()

    return {"class_id": prediction}
