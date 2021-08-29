import streamlit as st
import streamlit.components.v1 as stc
from PIL import Image
import librosa
import numpy as np
import pandas as pd
import os
import time
import plotly.express as px
import requests
import json


def init():
    global use_ploty, backend, print_time, recog_text
    use_ploty = True 
    print_time = False
    backend = "http://localhost:8000"

    st.set_page_config(layout="wide")
    return
    
def load_image(img_path):
    image = Image.open(img_path)
    return image

def get_emotion(uploaded_file):

    time.sleep(10)
    output_dic = {
        'Emotion':"Neutral",
        'Confidence':77,
        'probabilities':{
            'neutral':77,
            'joy':2,
            'anger':3,
            'disgust':5,
            'fear':2,
            'surprise':4,
            'sadness':9
        }
    }
    return output_dic

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_request(uploaded_file,col1):
 
    url = f"{backend}/get_predictions"

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    if uploaded_file.name.split(".")[-1]!="wav":
        with open(os.path.join("tmp","file.wav"), "wb") as f:
            f.write(uploaded_file.getvalue())
            f.close()
        #load the audio file
        y, sr = librosa.load(os.path.join("tmp","file.wav"))
        print(y.shape,sr,uploaded_file.name)
         #payload
        files = {"name":uploaded_file.name.split(".")[0]+".wav","y":y,"sr":sr}
        # delete the intermedite file
        #os.remove(os.path.join("tmp","file.wav"))
    else:
        #load the audio file
        y, sr = librosa.load(uploaded_file)
        print(y.shape,sr,uploaded_file.name)
        #payload
        files = {"name":uploaded_file.name,"y":y,"sr":sr}

    #post request
    response = requests.post(url, data=json.dumps(files,cls=NumpyEncoder))

    url1 = f"{backend}/speech_to_text"
    #post request
    response_text = requests.post(url1, data=json.dumps(files,cls=NumpyEncoder))
    print(response_text.json())

    if response_text.json()['text']!=' ':
        recog_text = response_text.json()['text']
        col1.markdown('<center><h3 style="color: black;">Recognized Text:'+recog_text+'</h3></center>',
                                unsafe_allow_html=True)

    return response.json()
    


def main():
    
    output_dic = {}
    #Header Image
    header = load_image(os.path.join("assets","header.png"))
    st.image(header,use_column_width=True)
    
    #Project Title
    st.markdown('<center><h1 style="color: black;">Audio Emotion Detection</h1></center>',
                            unsafe_allow_html=True)

    #layout
    au = st.empty()
    audio_container = au.container()
    start,col1, mid, col2 = au.columns([0.1,0.4,0.1,0.4])


    with col1:
        col1.markdown('<center><h3 style="color: black;">Input Audio</h3></center>',
                                unsafe_allow_html=True)

        uploaded_file = col1.file_uploader("Choose an Audio", type=["wav","mp3"])

        if uploaded_file is not None:
            col1.markdown('<center><h3 style="color: black;">Uploaded Audio</h3></center>',
                                unsafe_allow_html=True)
            col1.audio(uploaded_file)

                

    with col2:
        col2.markdown('<center><h3 style="color: black;">Prediction</h3></center>',
                                        unsafe_allow_html=True)
        if uploaded_file is not None:
            gif_path = os.path.join("assets","load.gif")
            gif_runner = col2.image(gif_path,use_column_width=True,width=10)
            
            output_dic = create_request(uploaded_file,col1)
            
            #output_dic = get_emotion(uploaded_file)
            gif_runner.empty()
            if output_dic['status']==200:
                if use_ploty:
                    df = pd.DataFrame.from_dict(output_dic['probabilities'],orient='index').reset_index().rename(columns={'index':'Labels',0:'Probability'})
                    df = df.sort_values(by=["Probability"], ascending=False)
                    df['Probability'] = df['Probability'].apply(lambda x: x*100)
                    colors = ['lightsalmon'] * 7
                    colors[0] = 'lightgreen'
                    fig = px.bar(df,x="Labels",y="Probability",orientation='v')
                    fig.update_traces(marker_color=colors)
                    col2.plotly_chart(fig, use_container_width=True)
                else:
                    df = pd.DataFrame.from_dict(output_dic['probabilities'],orient='index').rename(columns={'index':'Labels',0:'Probability'})
                    col2.bar_chart(df,use_container_width=True)
                col2.markdown('<center><h3 style="color: black;">Predicted Emotion: '+output_dic["Emotion"]+'</h3></center>',
                                unsafe_allow_html=True)

                col2.markdown('<center><h3 style="color: green;">Confidence Score: '+str(output_dic["Confidence"])+'%</h3></center>',
                                unsafe_allow_html=True)
                if print_time:
                    col2.markdown('<center><h3 style="color: green;">Execution Time: '+str(output_dic['execution_time'])+'sec</h3></center>',
                                unsafe_allow_html=True)
                    
            else:
                col2.markdown('<center><h3 style="color: red;">Error! Try other Audio</h3></center>',
                                unsafe_allow_html=True)
        



if __name__=='__main__':
    init()
    main()