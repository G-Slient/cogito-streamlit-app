import modules
import json
import requests
import librosa
import numpy as np
import os
import time
import soundfile as sf
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import Request
from pydantic import BaseModel

app = FastAPI()

model = modules.load_model()

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
    start_time = time.time()
    is_text = False

    y = np.asarray(item.y)
    sr = item.sr
    filename = item.name
    file_location = f"data/{filename}"
    sf.write(file_location,y,sr)
    print({"info": f"file '{filename}' saved at '{file_location}'"})

    

    #extract features from audio
    features_df = modules.audio_features(file_location)
    #speech to text
    if is_text:
        features_df['text'] = modules.audio_to_text(file_location)
    print(features_df.shape)
    #filesize
    features_df['filesize']= modules.calFileSize(file_location)
    #audio-duration
    features_df['fileduration']= modules.calAudioDuration(file_location)

    #features selection
    features = modules.load_features()

    output_dic = modules.getPredictions(model,features_df,features)
    print("Execution time {} sec::".format((time.time()-start_time)))
    output_dic['execution_time'] = round((time.time()-start_time),2)
    print(output_dic)
    return output_dic




@app.post("/speech_to_text")
async def speech_to_text(item: Item):
    
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    y = np.asarray(item.y)
    sr = item.sr
    filename = item.name
    file_location = f"tmp/{filename}"
    sf.write(file_location,y,sr)
    print({"info": f"file '{filename}' saved at '{file_location}'"})
    
    text = modules.audio_to_text(file_location)

    return {"text":text,"status":200}