#%%
import torch
import torch.optim as optim
import pandas as pd
from data_utils import train_valid_split, get_labels, load_dataset
from module import FocalLoss, train_E4, train_E7, train_full, validation, inference
from model import Efficientb4, Efficientb7
from simulation import seed_everything
import argparse
import warnings

warnings.filterwarnings(action = 'ignore') 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', default = 42, type = int, 
                        help = 'seed for repeatable results')
    parser.add_argument("--img_size", default = 380, type = int,
                        help = "input image size")  
    parser.add_argument("--epochs", default = 20, type = int,
                        help = "the number of training iteration")
    parser.add_argument("--batch_size", default = 16, type = int,
                        help = "batch size") 
    parser.add_argument("--lr", default = 5e-5, type = float,
                        help = "learning rate") 
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()

config = vars(get_args(debug = True))
seed_everything(config['seed']) # Seed 고정

df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_df, valid_df = train_valid_split(df, 0.8)
train_labels = get_labels(train_df)
valid_labels = get_labels(valid_df)

train_loader, valid_loader, test_loader = load_dataset(train_df, train_labels, valid_df, valid_labels, test_df, config['img_size'], config['batch_size'])

"""model : E4, img_size = 380, epoch = 20, batch = 16, seed = 42, answer20.csv"""
model = Efficientb4()
optimizer = optim.AdamW(params = model.parameters(), lr = config['lr'])
infer_model = train_E4(model, optimizer, config['epochs'], train_loader, valid_loader, device, loss = 'focal')

"""model : E7, img_size = 380, epoch = 20, batch = 8, Temperature = 0.1, seed = 42, answer27.csv"""
# model = Efficientb7(T = 0.1)
# optimizer = optim.AdamW(params = model.parameters(), lr = config['lr'])
# infer_model = train_E7(model, optimizer, config['epochs'], train_loader, valid_loader, device, loss = 'focal')

"""model : E7,  img_size = 256, epoch = 12, batch = 8, No validation set, seed = 40, answer45.csv"""
# model = Efficientb7()
# optimizer = optim.AdamW(params = model.parameters(), lr = config['lr'])
# infer_model = train_full(model, optimizer, config['epochs'], train_loader, device, loss = 'focal')

preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:,1:] = preds
submit.head()
submit.to_csv('./answer/answer20.csv', index = False)
