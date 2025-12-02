from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
from waterlevel import GaugeModel

app = FastAPI()

# load model once when server starts
model = GaugeModel("best.pt")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # save uploaded file temporarily
    file_name = f"temp_{uuid.uuid4()}.jpg"
    with open(file_name, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # run full pipeline
    result = model.process(file_name)
    return result
