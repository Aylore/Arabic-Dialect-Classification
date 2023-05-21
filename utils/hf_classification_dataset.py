from transformers.data.processors.utils import InputFeatures

from config import hf_model

from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, 
                          AutoTokenizer, BertTokenizer, Trainer,
                          )

class ClassificationDataset(Dataset):
    def __init__(self, text, target, model_name, max_len, label_map):
      super(ClassificationDataset).__init__()
      """
      Args:
      text (List[str]): List of the training text
      target (List[str]): List of the training labels
      tokenizer_name (str): The tokenizer name (same as model_name).
      max_len (int): Maximum sentence length
      label_map (Dict[str,int]): A dictionary that maps the class labels to integer
      """
      self.text = text
      self.target = target
      self.tokenizer_name = model_name
      self.tokenizer = AutoTokenizer.from_pretrained(hf_model.model_name)
      self.max_len = max_len
      self.label_map = label_map
      

    def __len__(self):
      return len(self.text)

    def __getitem__(self,item):
      text = str(self.text[item])
      text = " ".join(text.split())
        
      inputs = self.tokenizer(
          text,
          max_length=self.max_len,
          padding='max_length',
          truncation=True
      )      
      return InputFeatures(**inputs,label=self.label_map[self.target[item]])
    




def get_trainer_datasets(dataset):
    """
    Input : dataset from preprocess
    Output : datasets ready for training and validation
    """


    label_map = { v:index for index, v in enumerate(dataset.label_list) }
    # print(label_map)

    train_dataset = ClassificationDataset(
        dataset.train["text"].to_list(),
        dataset.train["label"].to_list(),
        hf_model.model_name,
        hf_model.max_len,
        label_map
    )
    test_dataset = ClassificationDataset(
        dataset.test["text"].to_list(),
        dataset.test["label"].to_list(),
        hf_model.model_name,
        hf_model.max_len,
        label_map
    )


    return train_dataset ,test_dataset