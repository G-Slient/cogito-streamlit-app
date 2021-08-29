# Cogito-Streamlit-App

<img src="frontend/assets/preview.gif">

## Instructions to set up the environment

1. Create a virtual environment

       python3 -m venv myenv

2. Activate the virtual envirnoment[Ubuntu]
      
       source myenv/bin/activate

3. Clone the Repo 

       git clone https://github.com/G-Slient/cogito-streamlit-app.git

4. Install the required packages

       pip install -r requirements.txt

## Instructions to execute the project

1. Start the backend fastapi server

       cd backend/  
       uvicorn main:app --reload  

2. Start the frontend Streamlit Server

       cd frontend/  
       streamlit run app.py

    Streamlit application can be accessed by 

       http://localhost:8501/