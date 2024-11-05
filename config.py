import torch
import argparse


class Config:
    def __init__(self,**kwargs):
        self.device = torch.device("cuda-0" if torch.cuda.is_available() else "cpu")
        for k,v in kwargs.items():
            setattr(self,k,v)
        

def get_config(**optional_kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode",type = str,default="train",help="Mode for the model [train|test]")
        parser.add_argument("--dataset_type",type = str,default = "TVSum",help = "choose the type of the dataset [TVSum|SumMe]")
        parser.add_argument("--num_epochs",type = int,required=True,help = "Mention the number of epochs")
        parser.add_argument("--batch_size",type = int,default = 1)
        parser.add_argument("--lr",type = float,required=True,help = "the learning rate of the model")
        parser.add_argument("--l2_reg",type = float,required=True,help = "the l2 regularization rate")
        parser.add_argument("--split_index",type = int,default = 0,help ="which split to run the model on")
        parser.add_argument("--num_blocks",type = int,required=True,help = "number of blocks(or graph nodes in the internal representation)")
        parser.add_argument("--embed_dim",type = int,default=1024,help = "dimension of feature embeddings")
        parser.add_argument("--num_heads",type = int,default=8,help = "number of heads in the multi head attention part of the model")
        parser.add_argument("--hidden_dim1",type = int,required = True)
        parser.add_argument("--hidden_dim2",type = int,required=True)
        parser.add_argument("--final_dim",type = int,required=True)
        parser.add_argument("--graph_dropout",type = float,required = True,help = "dropout rate for the graphNet")
        parser.add_argument("--threshold",type= int,default = 10,help = "temporal threshold between 2 connected nodes")
        args = parser.parse_args()
        kwargs = vars(args)
        kwargs.update(optional_kwargs)
        return Config(**kwargs)
    
