from model import SFCN
import loss as dpl
from utils import save_checkpoint, Dataset3d
import torch
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision.transforms import Resize, RandomRotation, ToTensor
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import pdb
from torch.utils.data.sampler import SubsetRandomSampler
import math
from sklearn.model_selection import train_test_split
import wandb
import argparse
from os.path import exists

parser = argparse.ArgumentParser(description='Arguments for model training.')
parser.add_argument('--sample_size', type=int, dest='sample_size', default=-1)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=8)

args = parser.parse_args()
BATCH_SIZE = args.batch_size
EPOCH = 130
LEARNING_RATE = 0.01
SAMPLE_SIZE = args.sample_size
wandb_run_name = 'BATCHSIZE'+str(BATCH_SIZE)+'_EPOCH'+str(EPOCH)+'_SAMPLESIZE'+str(SAMPLE_SIZE)
wandb.init(project="brain_age_regression",name=wandb_run_name)
wandb.config.batch_size = BATCH_SIZE
wandb.config.epoch = EPOCH
wandb.config.sample_size = SAMPLE_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


subject_images_days_df = pd.read_csv('../../datasets/ukbb/metadata/subjets_images_dayssincebaseline.csv').dropna()
subject_images_days_df['age_at_image'] = (subject_images_days_df['days_since_baseline']/365)+subject_images_days_df['age_at_baseline']
# When subjects have more than one image, just keep one, CHECK if first image
subject_images_days_df.drop_duplicates('subject_id', inplace = True)
subject_images_days_df.reset_index(inplace=True)

#Sample dataset to speed up training
if SAMPLE_SIZE == -1:
    subject_images_days_df = subject_images_days_df
else:
    subject_images_days_df = subject_images_days_df.sample(n=SAMPLE_SIZE)
print(len(subject_images_days_df))
validation_split = .2
test_split = .1
shuffle_dataset = True
random_seed= 42

subject_images_days_train_df, subject_images_days_remain_df = train_test_split(subject_images_days_df,
                                                          test_size=(validation_split + test_split),shuffle=shuffle_dataset,random_state=random_seed)
subject_images_days_validate_df, subject_images_days_test_df  = train_test_split(subject_images_days_remain_df,
                                                          train_size=(1/(validation_split + test_split))*validation_split,
                                        test_size=(1/(validation_split + test_split))*test_split,shuffle=shuffle_dataset,random_state=random_seed)




train_dataset = Dataset3d(subject_images_days_train_df,test=False)
validate_dataset = Dataset3d(subject_images_days_validate_df,test=True)
test_dataset = Dataset3d(subject_images_days_test_df,test=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
validation_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=1)
model = SFCN()
model = torch.nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
saved_model_name = '{0}_checkpoint.pth.tar'.format(wandb_run_name)
if exists(saved_model_name):
    print('RESUMING TRAINING')
    checkpoint = torch.load(saved_model_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_mae = checkpoint['best_mae']
    start_epoch_range = checkpoint['epoch']
    model.train()
else:
    best_mae = 999999
    start_epoch_range = 0

loss_func = dpl.my_KLDivLoss



wandb.watch(model)
running_loss = 0
for epoch in range(start_epoch_range,EPOCH):
    if epoch % 30 == 0:
        if epoch != 0:
            LEARNING_RATE = LEARNING_RATE*0.3
        print(LEARNING_RATE)
    for step, (batch_x, batch_y) in enumerate(train_loader):  # for each training step
        prediction = model(batch_x)  # input x and predict based on x


        loss = loss_func(prediction[0].reshape([len(batch_y[0]),-1]), batch_y[0])  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        running_loss += loss.item()
        if step % 10 == 0:
            print ('epoch [{}], Loss: {:.2f}'.format(epoch, loss.item()))
            wandb.log({"loss": loss,"epoch":epoch})
    running_mae = 0
    running_mse = 0
    model.eval()
    with torch.no_grad():
        for step, (batch_val_x, batch_val_y) in enumerate(validation_loader):
            prediction = model(batch_val_x)
            age_bin_centers = batch_val_y[1][0]
            prediction_reshaped = prediction[0].reshape([1,-1]).reshape(-1)
            prediction_age = torch.exp(prediction_reshaped)@age_bin_centers
            y_reshaped = batch_val_y[0].reshape(-1)
            y_age = y_reshaped @ age_bin_centers
            error = torch.abs(prediction_age - y_age).sum()
            squared_error = ((prediction_age - y_age) * (prediction_age - y_age)).sum()
            running_mae += error
            running_mse += squared_error
    mse = math.sqrt(running_mse/len(validation_loader))
    mae = running_mae/len(validation_loader)
    training_loss = running_loss/len(train_loader)
    with open(wandb_run_name +'_training_results.csv', 'a') as filedata:
        fieldnames = ['epoch', 'training_loss', 'mae','mse']
        writer = csv.DictWriter(filedata, delimiter=',', fieldnames=fieldnames)
        writer.writerow({'epoch': epoch, 'training_loss': training_loss,
                         'mae':mae,'mse':mse})
        # remember best acc@1 and save checkpoint
    is_best = mae < best_mae
    best_mae = min(mae, best_mae)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_mae': best_mae,
        'optimizer': optimizer.state_dict(),
    }, is_best,filename='{0}_checkpoint.pth.tar'.format(wandb_run_name),best_file_name='{0}_model_best.pth.tar'.format(wandb_run_name))
    print ('epoch [{}], MAE: {:.2f}'.format(epoch, mae))
    wandb.log({"validation_mse": mse,"validation_mae": mae,"epoch":epoch})
    model.train()


