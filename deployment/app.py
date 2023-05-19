from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np


app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post(  "/predict")
async def predict(request: Request):
    form_data = await request.form()
    text = form_data["text"]

    label = predict_label(text)
    
    return templates.TemplateResponse("prediction.html", {"request": request, "label": label})


def predict_label(text):
    # Dummy prediction logic
    model_prediction = np.random.choice(["SD","LB","EG","MA", "LY"]) #Put model here
    
    dict_ = {"SD":"Sudanese - سودانية",
             "LB":"Libyan - ليبية",
             "EG":"Egyptian - مصرية",
             "MA":"moroccian - مغربية",
             "LY":"Lebanese -لبنانية"}
    
    return dict_[model_prediction]

# to run the app: uvicorn app:app --reload
