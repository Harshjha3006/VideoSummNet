import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F

from torch_geometric.data import Data

"""
for 1 video 
x -> (n_segs,n_frame_per_seg,feature_dim)
passed through mha
out (feature vectors which are aware of their respective segment)
res (feature vector of each segment)

x -> (n_segs,feature_dim of each segment)
passed through graphNet
res (feature vectors of segments which are aware of other segments)

for each frame 
final_res = x + out + res
final_res -> importance score 

"""

from model.attention import MultiHeadAttention
from model.graphNet import GraphNet

def build_edges(num_segs,threshold):

    edge_source,edge_dest,edge_attr = [],[],[]

    for i in range(num_segs):
        for j in range(num_segs):
            if abs(i - j) <= threshold:
                edge_source.append(i)
                edge_dest.append(j)
                edge_attr.append(np.sign(i - j))

    return edge_source,edge_dest,edge_attr


def build_graph(block_outputs,threshold):
    edge_source,edge_dest,edge_attr = build_edges(block_outputs.shape[1],threshold)
    return Data(x = block_outputs,
                edge_index = torch.tensor(np.array([edge_source,edge_dest],dtype = np.int64),dtype = torch.long),
                edge_attr = torch.tensor(edge_attr,dtype = torch.long))
    


class SummaryNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_blocks = config.num_blocks
        self.threshold = config.threshold
        self.mha = MultiHeadAttention(config.embed_dim,config.num_heads)
        self.graphnet = GraphNet(config.embed_dim,config.hidden_dim1,config.hidden_dim2,config.final_dim,config.graph_dropout)
        self.linear = nn.Linear(config.embed_dim,1)


    def forward(self,x,y):
        batch_size,seq_len,embed_dim = x.shape
        block_size = seq_len // self.num_blocks
        new_seq_len = block_size * self.num_blocks
        x = x[:,:new_seq_len,:]
        y = y[:,:new_seq_len,:]
        x = torch.reshape(x,(batch_size,self.num_blocks,block_size,embed_dim))

        block_outputs = []
        block_embeddings = []
        for i in range(self.num_blocks):
            block = x[:,i,:,:]
            # (be -> (b,bsize,1024))
            block_embedding,block_output = self.mha(block)
            block_outputs.append(block_output)
            block_embeddings.append(block_embedding)
        # (b,num_blocks,1024)
        block_outputs = torch.stack(block_outputs,dim = 1)
        # (b,num_blocks,bsize,1024)
        block_embeddings = torch.stack(block_embeddings,dim = 1) 

        with torch.no_grad():
            graph_data = build_graph(block_outputs,self.threshold)
        
        # (b,num_blocks,1024)
        graph_output = self.graphnet(graph_data.x,graph_data.edge_index,graph_data.edge_attr)
        graph_output = torch.unsqueeze(graph_output,dim = 2).expand(batch_size,self.num_blocks,block_size,embed_dim)

        out = x + block_embeddings + graph_output
        out = torch.reshape(out,(batch_size,self.num_blocks * block_size,embed_dim))
        out = self.linear(out)
        out = F.softmax(out,dim = -1)
        return out,y


        

        











        



        