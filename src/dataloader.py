from torch.utils.data import Dataset
import pandas as pd
import pdb
import numpy as np


class TripletDataset(Dataset): ## TODO: Add deep_ct
    def __init__(self, 
        filepath):
        """
        Args: 
            filepath (string): address of the the query and passage document for training
        """
        self.df = pd.read_parquet(filepath) 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pos_sentence = self.df.iloc[idx].abstract_x
        query_sentence = self.df.iloc[idx].query_text
        neg_sentence = self.df.iloc[idx].abstract_y
        pos_biggraph = self.df.iloc[idx].biggraph_embedding_x
        neg_biggraph = self.df.iloc[idx].biggraph_embedding_y

        sample = {
                    'query_sentence': query_sentence,
                    'pos_sentence': pos_sentence,
                    'neg_sentence': neg_sentence,
                    'pos_biggraph': pos_biggraph,
                    'neg_biggraph': neg_biggraph
                    }
        return sample

if __name__=='__main__':
    data = TripletDataset('../data/processed/folds/fold_0_test.parquet.gzip')
    print(data[0])

