

from config import hf_model

from transformers import AutoModelForSequenceClassification , Trainer

from utils import (fetch_data , split_dataset_hf  ,custom_dataset_hf,
                   hf_preprocess , hf_classification_dataset , metrics)



df = fetch_data.get_data()

df_train , df_test = split_dataset_hf.split_hf(df)   ## split data for train and test

dataset = custom_dataset_hf.get_dataset(df_train , df_test , df)  ## create a custom dataset instance


preprocessed_df = hf_preprocess(dataset) ## process text in the dataset


train_dataset , test_dataset = hf_classification_dataset.get_trainer_datasets(preprocessed_df)







def model_init():
    return AutoModelForSequenceClassification.from_pretrained(hf_model.model_name,
                                                               return_dict=True,
                                                                 num_labels=5)



trainer = Trainer(
    model = model_init(),
    args = hf_model.training_args,
    train_dataset = train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=metrics.compute_metrics,
)