from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np


app = FastAPI()


app.mount("/static", StaticFiles(directory="deployment/static"), name="static")
templates = Jinja2Templates(directory="deployment/templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def predict(request: Request):
    form_data = await request.form()
    text = form_data["text"]

    label, SD_score, LB_score, EG_score, MA_score, LY_score = predict_label(text)
    return templates.TemplateResponse("index.html",
                                      {"request": request,
                                       "label": label,
                                       "text": text,
                                       "SD_score" : "{:.1f}%".format(SD_score), 
                                       "LB_score" : "{:.1f}%".format(LB_score), 
                                       "EG_score" : "{:.1f}%".format(EG_score), 
                                       "MA_score" : "{:.1f}%".format(MA_score), 
                                       "LY_score" : "{:.1f}%".format(LY_score), 
                                       "eg_width" : f"width: {EG_score}%;",
                                       "lb_width" : f"width: {LB_score}%;",
                                       "ly_width" : f"width: {LY_score}%;",
                                       "ma_width" : f"width: {MA_score}%;",
                                       "sd_width" : f"width: {SD_score}%;",
                                       })

@app.get("/clear/")
async def clear(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "text": ""})

def predict_label(text):
    # Dummy prediction logic
    model_prediction = np.random.choice(["SD","LB","EG","MA", "LY"]) #Put model here
    
    dict_ = {"SD":"Sudanese - سودانية",
             "LB":"Libyan - ليبية",
             "EG":"Egyptian - مصرية",
             "MA":"moroccian - مغربية",
             "LY":"Lebanese -لبنانية"}
    
    SD_score = np.random.rand() * 100
    LB_score = np.random.rand() * 100
    EG_score = np.random.rand() * 100
    MA_score = np.random.rand() * 100
    LY_score = np.random.rand()* 100

    return dict_[model_prediction], SD_score, LB_score, EG_score, MA_score, LY_score

# to run the app: uvicorn deployment.app:app --reload
