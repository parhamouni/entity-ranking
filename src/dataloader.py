from torch.utils.data import Dataset
import pandas as pd
import pdb
import numpy as np


class TripletDataset(Dataset): ## TODO: Add deep_ct
    def __init__(self, 
        filepath, biggraph = False, deepct = False):
        """
        Args: 
            filepath (string): address of the the query and passage document for training
        """
        self.df = pd.read_parquet(filepath) 
        self.biggraph = biggraph
        self.deepct = deepct

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {  'query_sentence': self.df.iloc[idx].query_text,
                    'pos_sentence': self.df.iloc[idx].abstract_x,
                    'neg_sentence': self.df.iloc[idx].abstract_y}
        if self.biggraph:
            sample['pos_biggraph'] = self.df.iloc[idx].biggraph_embedding_x
            sample['neg_biggraph'] = self.df.iloc[idx].biggraph_embedding_y
        if self.deepct:
            sample['pos_deepct'] = np.nan_to_num(self.df.iloc[idx].deepct_weights_x)
            sample['neg_deepct']  = np.nan_to_num(self.df.iloc[idx].deepct_weights_y)
        return sample

if __name__=='__main__':
    data = TripletDataset('../data/processed/folds/fold_0_test.parquet.gzip',biggraph = True, deepct = True)
    sample = data[0]
    pdb.set_trace()

