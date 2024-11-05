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


segs = []

for i, j in change_points : 
    features = frame_features[i:j]



"""

from config import get_config
from dataloader import get_data_loader
from model.summaryNet import SummaryNet
import torch
import torch.nn as nn

def train(config):
    torch.autograd.set_detect_anomaly(True)
    model = SummaryNet(config).to(config.device)
    optimiser = torch.optim.Adam(model.parameters(recurse = True),lr = config.lr,weight_decay=config.l2_reg)
    train_loader = get_data_loader(config)
    num_batches = int(len(train_loader) / config.batch_size)
    loss_history = []
    criterion = nn.MSELoss()
    for epoch in range(config.num_epochs):
        for _ in range(num_batches):
            iterator = iter(train_loader)
            optimiser.zero_grad()
            data = next(iterator)
            x = data["features"].to(config.device)
            y = data["gtscore"].to(config.device)
            preds,y = model(x,y)
            loss = criterion(preds.squeeze(0),y.squeeze(0))
            loss.backward()
            loss_history.append(loss.data)
        print(f"Epoch {epoch} loss : {torch.stack(loss_history).mean()}")



if __name__ == "__main__":
    config = get_config()
    train(config)
    