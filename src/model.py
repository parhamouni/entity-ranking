from sentence_transformers import models,SentenceTransformer
from torch import nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from sentence_transformers.util import batch_to_device
from utils import TripletLoss
from nfnets import SGD_AGC
import pdb


class Net(pl.LightningModule):
    """
    Parameters
    ----------
    transformer_name: name of the backbone architecture, for example nlpaueb/legal-bert-base-uncased
    learning_rate: learning rate
    """
    def __init__(self, transformer_name,big_graph, deep_ct,learning_rate):
        super(Net, self).__init__()
        # lightning initial configs
        self.save_hyperparameters()
        self.lr = learning_rate
        self.big_graph = big_graph
        self.deep_ct = deep_ct
        self.additional_dim = 0

        # self.retrieval_mode = retrieval_mode
        # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
        word_embedding_model = models.Transformer(transformer_name)
        self.lm_dim = word_embedding_model.get_word_embedding_dimension()
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(self.lm_dim,
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        self.lm_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        if self.big_graph:
            self.additional_dim +=200 ## dimension of biggraph embeddings
        if deep_ct:
            self.additional_dim +=100 ## dimension of deepct weights to be checked

        self.linear_p1 = nn.Sequential(
            nn.Linear(self.lm_dim+self.additional_dim, self.lm_dim)
        )
        self.linear_p2 = nn.Sequential(
            nn.Linear(self.lm_dim, self.lm_dim)
        )
        self.loss_criteria = TripletLoss()


    def forward(self,batch):
        q_rep =self.lm_model(batch['query_sentence'])['sentence_embedding']
        p_rep = self.lm_model(batch['pos_sentence'])['sentence_embedding']
        n_rep = self.lm_model(batch['neg_sentence'])['sentence_embedding']
        if self.big_graph:
            pos_biggraph = batch['pos_biggraph']
            neg_biggraph = batch['neg_biggraph']
            p_rep = torch.cat([p_rep,pos_biggraph], dim=1)
            n_rep = torch.cat([n_rep,neg_biggraph], dim=1)
        if self.deep_ct:
            pos_deepct = batch['pos_deepct']
            neg_deepct = batch['neg_deepct']
            p_rep = torch.cat([p_rep,pos_deepct], dim=1)
            n_rep = torch.cat([n_rep,neg_deepct], dim=1)

        p_rep = p_rep.float()
        n_rep= n_rep.float()
        p_rep = F.relu(self.linear_p1(p_rep))
        n_rep = F.relu(self.linear_p1(n_rep))
        p_rep =self.linear_p2(p_rep)
        n_rep =self.linear_p2(n_rep)
        return q_rep, p_rep,n_rep
    

    def collate_fn(self,batch):
        device = 'cuda' if 'cuda' in self.lm_model.device.__str__() else 'cpu'
        for k in batch.keys():
            if any(item in k for item in ['deepct', 'biggraph']):
                batch[k] = torch.tensor(batch[k]).to(device)
            else:
                batch[k] = batch_to_device(self.lm_model.tokenize(batch[k]),device)
        return batch
    

    def training_step(self, batch, batch_idx):
        batch = self.collate_fn(batch)
        q_rep, p_rep,n_rep = self.forward(batch) 
        loss = self.loss_criteria(q_rep, p_rep,n_rep)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.collate_fn(batch)
        q_rep, p_rep,n_rep = self.forward(batch)
        loss = self.loss_criteria( q_rep, p_rep,n_rep)
        self.log('val_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = SGD_AGC(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 1),
            'name': 'lr_logger','monitor': 'train_loss'
        }
        return {
               'optimizer': optimizer,
               'lr_scheduler': lr_scheduler,
               'monitor': 'train_loss'
           }
