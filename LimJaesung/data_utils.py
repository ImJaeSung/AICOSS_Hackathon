import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def train_valid_split(data, train_len : float):
    data = data.sample(frac = 1)
    train_len = int(len(data) * train_len)
    train_df = data[:train_len]
    valid_df = data[train_len:]

    return train_df, valid_df

def get_labels(df):              
    return df.iloc[:,2:].values

class Mixup:
    def __init__(self, target, scale, alpha = 0.5, beta = 0.5):
        self.target = target
        self.scale = scale
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image):
        image = np.array(image)
        target = self.target.resize(self.scale)
        target = np.array(target)
        mix_image = image*self.alpha + target*self.beta
        return Image.fromarray(mix_image.astype(np.uint8))

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transform = None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        # PIL 이미지로 불러오기
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        if self.label_list is not None:
            label = torch.tensor(self.label_list[index], dtype = torch.float32)
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)

def load_dataset(train_df, train_labels, valid_df, valid_labels, test_df, img_size, batch_size):    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size),
                          interpolation = transforms.InterpolationMode.BICUBIC),
        Mixup(target = train_labels,)
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), 
                          interpolation = transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(train_df['img_path'].values, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

    valid_dataset = CustomDataset(valid_df['img_path'].values, valid_labels, test_transform)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
    
    test_dataset = CustomDataset(test_df['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
    
    return train_loader, valid_loader, test_loader

class Mixup:
    def __init__(self, target, scale, alpha = 0.5, beta = 0.5):
        self.target = target
        self.scale = scale
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image):
        image = np.array(image)
        target = self.target.resize(self.scale)
        target = np.array(target)
        mix_image = image*self.alpha + target*self.beta
        return Image.fromarray(mix_image.astype(np.uint8))