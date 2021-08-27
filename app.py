import streamlit as st
import streamlit.components.v1 as stc
from PIL import Image
import librosa
import numpy as np
import pandas as pd
import os
import time
import plotly.express as px

def init():
    global use_ploty
    use_ploty = False 

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
            output_dic = get_emotion(uploaded_file)
            gif_runner.empty()
            if output_dic['probabilities']:
                if use_ploty:
                    df = pd.DataFrame.from_dict(output_dic['probabilities'],orient='index').reset_index().rename(columns={'index':'Labels',0:'Probability'})
                    fig = px.bar(df,y="Labels",x="Probability",orientation='h')
                    col2.plotly_chart(fig, use_container_width=True)
                else:
                    df = pd.DataFrame.from_dict(output_dic['probabilities'],orient='index')
                    col2.bar_chart(df,use_container_width=True)
                col2.markdown('<center><h3 style="color: black;">Predicted Emotion: '+output_dic["Emotion"]+'</h3></center>',
                                unsafe_allow_html=True)

                col2.markdown('<center><h3 style="color: green;">Confidence Score: '+str(output_dic["Confidence"])+'%</h3></center>',
                                unsafe_allow_html=True)

            


if __name__=='__main__':
    init()
    main()