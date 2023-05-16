
from typing import List
from tqdm import tqdm_notebook as tqdm
import pandas as pd

class CustomDataset:
    def __init__(
        self,
        name: str,
        train: List[pd.DataFrame],
        test: List[pd.DataFrame],
        label_list: List[str],
    ):
        """Class to hold and structure datasets.

        Args:

        name (str): holds the name of the dataset so we can select it later
        train (List[pd.DataFrame]): holds training pandas dataframe with 2 columns ["text","label"]
        test (List[pd.DataFrame]): holds testing pandas dataframe with 2 columns ["text","label"]
        label_list (List[str]): holds the list  of labels
        """
        self.name = name
        self.train = train
        self.test = test
        self.label_list = label_list




def get_dataset(df_train  ,df_test ,df):

    """
    Input : output of split_dataset_tf and the orginal df 
    Output : dataset class for train and test
    """
    dataset = CustomDataset("Arabic_dialect" , df_train , df_test , df["dialect"].unique().tolist())
    return dataset