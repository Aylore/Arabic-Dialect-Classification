from transformers import AutoModelForSequenceClassification , AutoTokenizer
from config import hf_model
import numpy as np
from utils import ml_preprocessing 

from transformers import TextClassificationPipeline


label_map_list = list(hf_model.label_map.keys())

model_path = "/models/checkpoint-3924"


model = AutoModelForSequenceClassification.from_pretrained(model_path , return_dict = True , num_labels = len(hf_model.label_map) )

tok = AutoTokenizer.from_pretrained(hf_model.model_name)

pipe = TextClassificationPipeline(model=model, tokenizer=tok, return_all_scores=True)


def predict(tweet  : str , return_scores = False ): 
    
    text = ml_preprocessing.wrangle_ml(tweet)

    scores  = pipe(text)
    scores = np.array(scores)

    dialect_pred =  label_map_list[scores.argmax()]

    
    if return_scores:
        scores_dict = dict(zip(label_map_list,scores))
        return dialect_pred , scores_dict
    else :
        return dialect_pred

