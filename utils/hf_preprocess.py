from arabert.preprocess import ArabertPreprocessor
from config import hf_model






def process_text(dataset):
    """
    Input : custom dataset instances
    Output : processed dataset ready for clssification dataset
    """

    arabic_prep = ArabertPreprocessor(hf_model.model_name)

    dataset.train["text"] = dataset.train["text"].apply(lambda x: arabic_prep.preprocess(x))
    dataset.test["text"] = dataset.test["text"].apply(lambda x: arabic_prep.preprocess(x))  

    return dataset