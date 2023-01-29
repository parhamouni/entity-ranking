from argparse import ArgumentParser
from dataloader import TripletDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from model import Net
import warnings
import logging
import pdb
import os
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)


def cli_main():
    pl.seed_everything(1234)
    # ------------
    # args
    # ------------
    parser = ArgumentParser(description="Fine-tune variants of siamese encoder architectures")
    parser.add_argument('--batch_size', default=10, type=int,help='batch_size')
    parser.add_argument('--learning_rate', default=0.1, type=float,help='Learning rate')
    parser.add_argument('--biggraph_embedding', action='store_true', default = False  ,help='Use biggraph embeddings')
    parser.add_argument('--deepct', action='store_true', default = False  ,help='Use deepct embeddings and weights')
    parser.add_argument('--transformer_name', type=str, default='microsoft/mpnet-base')
    parser.add_argument('--num_workers', default=100, type=int,help='num_workers in dataloader')
    parser.add_argument('--training_file_address', type=str,
                        default="../data/processed/folds/fold_0_training.parquet.gzip",
                        help= "address of the training path")
    parser.add_argument('--testing_file_address',
                        default='../data/processed/folds/fold_0_test.parquet.gzip',
                        type=str,
                        help='address of the testing path')
    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    # ------------
    # model
    # ------------
    model = Net(args.transformer_name,
                args.biggraph_embedding,
                args.deepct,
                args.learning_rate)
    # ------------
    # training
    # ------------
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor = 'val_loss')
    trainer = pl.Trainer.from_argparse_args(args, accelerator="auto",
                    callbacks=[lr_monitor,
                    checkpoint_callback,
                    EarlyStopping('val_loss',patience = 5)])
    logging.info('Loading training triplet dataset')
    train_dataset = TripletDataset(filepath = args.training_file_address,
                                    biggraph = args.biggraph_embedding,
                                     deepct = args.deepct)
    logging.info('Loading testing triplet dataset')
    val_dataset = TripletDataset(filepath=args.testing_file_address,
                                biggraph = args.biggraph_embedding,
                                deepct = args.deepct)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    cli_main()

