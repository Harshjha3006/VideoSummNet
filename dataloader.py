import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import json
import numpy as np

class VideoFramesData(Dataset):
    def __init__(self,mode,dataset_type,split_index):
        self.datasets = ["datasets/eccv16_dataset_summe_google_pool5.h5",
                         "datasets/eccv16_dataset_tvsum_google_pool5.h5"]
        
        self.split_files = ['splits/summe_splits.json','splits/tvsum_splits.json']
        self.dataset,self.split_file,self.split = None,None,None
        if dataset_type == "TVSum":
            self.dataset = self.datasets[1]
            self.split_file = self.split_files[1]
        else:
            self.dataset = self.datasets[0]
            self.split_file = self.split_files[0]

        h5file = h5py.File(self.dataset,'r')

        self.frame_info = []
        with open(self.split_file,'r') as f:
            data = json.loads(f.read())
            for i,split in enumerate(data):
                if i == split_index:
                    self.split = split
                    break
        
        for video_name in self.split[mode + "_keys"]:
            features = torch.tensor(np.array(h5file[video_name + "/features"]),dtype = torch.float32)
            gtscore = torch.tensor(np.array(h5file[video_name + "/gtscore"]),dtype = torch.float32).unsqueeze(1)
            self.frame_info.append({'features' : features,'gtscore' : gtscore})

        h5file.close()

    def __len__(self):
        return len(self.frame_info)
    
    def __getitem__(self,index):
        return self.frame_info[index]
    



def get_data_loader(config):
    vd = VideoFramesData(config.mode,config.dataset_type,config.split_index)
    return DataLoader(vd,batch_size = config.batch_size,shuffle=True)