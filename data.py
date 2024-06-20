import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os.path as osp

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


class Phase_Video_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, gt_dir, feat_dir,video_IDs, dataset):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gt_dir = gt_dir
        self.feat_dir = feat_dir
        self.video_IDs = video_IDs
        
        if dataset == "cholec80":
            self.action_dict = {'Preparation': 0, 'CalotTriangleDissection': 1, 'ClippingCutting': 2,\
                            'GallbladderDissection': 3, 'GallbladderPackaging': 4, \
                            'CleaningCoagulation': 5, 'GallbladderRetraction': 6}

        elif dataset == "m2cai":
            self.action_dict = {'TrocarPlacement': 0, 'Preparation': 1, 'CalotTriangleDissection': 2, 'ClippingCutting': 3,\
                    'GallbladderDissection': 4, 'GallbladderPackaging': 5, \
                    'CleaningCoagulation': 6, 'GallbladderRetraction': 7}

    def __len__(self):
        return len(self.video_IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        video_index = self.video_IDs[idx]
        video_name = "video"+str(video_index+1).zfill(2)
        
        features = np.load(osp.join(self.feat_dir, video_name + '.npy'))
        features = torch.tensor(features, dtype=torch.float)
        
        gt_file = osp.join(self.gt_dir, video_name + ".txt")
        phase = read_file(gt_file).split('\n')[0:-1]
        phase = [self.action_dict[i] for i in phase]
        phase = torch.LongTensor(phase)
        return features, phase


def create_dataloader(gt_dir, feat_dir, dataset, phase):
    
    partition = dict()
    
    if phase == "train":
        if dataset == "cholec80":
            partition["train"] = list(range(32))
            partition["val"] = list(range(32,40))
        
        elif dataset == "m2cai":
            partition["train"] = list(range(20))
            partition["val"] = list(range(20,27))
            
    elif phase == "finetune":
        if dataset == "cholec80":
            partition["train"] = list(range(40))
            partition["val"] = list(range(32,40))
    
        elif dataset == "m2cai":
            partition["train"] = list(range(27))
            partition["val"] = list(range(20,27))

    train_dataset = Phase_Video_Dataset(gt_dir=gt_dir, feat_dir=feat_dir, video_IDs = partition['train'], dataset=dataset)
    val_dataset = Phase_Video_Dataset(gt_dir=gt_dir, feat_dir=feat_dir, video_IDs = partition['val'], dataset=dataset)
   
    dataset_size_train = len(train_dataset)
    dataset_size_val = len(val_dataset)

    train_dataloader =  DataLoader(train_dataset, batch_size=1,shuffle=True, num_workers=20, pin_memory=True)
    val_dataloader =  DataLoader(val_dataset, batch_size=1,shuffle=True, num_workers=20, pin_memory=True)

    print("TrainSet contains {} videos.".format(dataset_size_train))
    print("ValSet contains {} videos.".format(dataset_size_val))

    return train_dataloader, val_dataloader, dataset_size_train, dataset_size_val

