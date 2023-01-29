import pandas as pd
from argparse import ArgumentParser
import numpy as np
import logging
from dataloader import QueryPassageDataset
import pytorch_lightning as pl
from model import Net
import torch
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
import pdb
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)

EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)


def scorer(trainer,model,table_address,batch_size,num_workers,exp_name, biggraph,deepct  ):
    dataset = QueryPassageDataset(filepath = table_address,biggraph = biggraph,deepct = deepct  )
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True,num_workers = num_workers) 
    if 'score_' + exp_name in dataset.df.columns:
        logging.info('Scores are already available')
    else:
        logging.info('Embeddings inference started')            
        embeddings= trainer.predict(model, dataloader)
        q_rep = []
        p_rep = []
        for x in embeddings:
            q_rep.append(x[0])
            p_rep.append(x[1])
        # pdb.set_trace()
        q_rep = torch.cat(q_rep)
        p_rep = torch.cat(p_rep)
        # dataset.df['query_emb_' + exp_name] = q_rep
        # dataset.df[ 'passage_emb_' + exp_name] = p_rep
        logging.info('Embeddings inferred')
        logging.info('Calculate scores')
        sim_scores = EUCLIDEAN(q_rep,p_rep)
        sim_scores = sim_scores.numpy()
        dataset.df['score_' + exp_name] = sim_scores
        dataset.df.to_parquet(table_address, compression='gzip')
        logging.info('Score generation finished for {} and table {}'.format(exp_name, table_address))
        logging.info('*'*150)






def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--table_address', type=str,help='Address of the memo tables that contain the query sentences')
    parser.add_argument('--batch_size', default=10000, type=int,help='batch_size')
    parser.add_argument('--num_workers', default=50, type=int,help='num_workers in dataloader')
    parser.add_argument('--gpu', default=1, type=int,help='gpu address')
    parser.add_argument('--model_name', type=str,help='a model address that contains a .ckpt or an off-the-shelf-model name')
    args = parser.parse_args()

    ## loading the model

    if ".ckpt" in args.model_name:
        logging.info('Load the model from the checkpoint {}'.format(args.model_name))
        model = Net.load_from_checkpoint(args.model_name)

    trainer = pl.Trainer(gpus=1,logger=False)

    exp_name =  model.hparams["transformer_name"] + '_' + str( model.hparams["big_graph"]) + '_' + str(model.hparams["deep_ct"])

    scorer(trainer,model,args.table_address,args.batch_size,args.num_workers,exp_name, 
    model.hparams["big_graph"],model.hparams["deep_ct"])







if __name__ == '__main__':
    cli_main()

