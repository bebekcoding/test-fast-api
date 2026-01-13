from fastapi import FastAPI
from model.simple_model import MyModel
import torch
from typing import List
from utils import config
from pathlib import Path

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

    return post_processing(prediction)

@myapp.get("/health")
def check_health():
    checkpoint_path = "checkpoints\checkpoints-1000.pth"

    checkpoint_file = Path(checkpoint_path)

    if checkpoint_file.is_file():
        return {"is_checkpoint_exist": True}
    else:
        return {"is_checkpoint_exist": False}



def post_processing(class_id: int) -> dict:
    if class_id == 0:
        return {"class_name": "anjing"}
    elif class_id == 1:
        return {"class_name": "kucing"}
    elif class_id == 2:
        return {"class_name": "burung"}

