from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset', 'AuGDataset']

class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_data, video_data, audio_data, other_data):
        
        self.label_ids = label_ids
        self.text_data = text_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.size = len(self.text_data)

        self.other_data = other_data
        if self.other_data is not None:
            for key in other_data.keys():
                setattr(self, key, other_data[key])  
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_data[index]),
            'video_feats': torch.tensor(np.array(self.video_data['feats'][index])),
            'video_lengths': torch.tensor(np.array(self.video_data['lengths'][index])),
            'audio_feats': torch.tensor(np.array(self.audio_data['feats'][index])),
            'audio_lengths': torch.tensor(np.array(self.audio_data['lengths'][index])),
        }

        if self.other_data is not None:    
            for key in self.other_data.keys():
                sample[key] = torch.tensor(getattr(self, key)[index])
        
        return sample


class AuGDataset(Dataset):
        
    def __init__(self, label_ids, text_feats):
        
        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': self.label_ids[index], 
            'text_feats': self.text_feats[index],
        } 
        return sample