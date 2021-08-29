import modules
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import Request
from pydantic import BaseModel
import json,requests
import librosa
import numpy as np
import soundfile as sf

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/upload")
async def get_audio_file(file: UploadFile = File(...)):
    file_location = f"data/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    print({"info": f"file '{file.filename}' saved at '{file_location}'"})
    features_df = modules.audio_features(file_location)
    features_df['text'] = modules.audio_to_text(file_location)
    print(features_df)
    return "Done"




class Item(BaseModel):
    name: str
    y: list
    sr: int

@app.post("/get_predictions")
async def get_predictions(item: Item):
    
    y = np.asarray(item.y)
    sr = item.sr
    filename = item.name
    file_location = f"data/{filename}"
    
    sf.write(file_location,y,sr)
    print({"info": f"file '{filename}' saved at '{file_location}'"})
    features_df = modules.audio_features(file_location)
    features_df['text'] = modules.audio_to_text(file_location)
    print(features_df.shape)

    return "Success"