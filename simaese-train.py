#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class DataPrep:
    def __init__(self):
        self.min_lat = 22.502
        self.max_lat = 22.751
        self.min_long = 113.813
        self.max_long = 114.293
        self.max_grid_long = 48
        self.max_grid_lat = 25
        self.csv_data_dir = "./csv_files"
        self.grid_data_dir = "./grid_data"
        create_folder(self.csv_data_dir)
        create_folder(self.grid_data_dir)
        # create_folder(self.test_csv_dir)

    def create_csv(self):
        with open("./project_4_train.pkl", 'rb') as f:
            data_dict = pickle.load(f)
        f.close()
        columns = ["plate", "longitude", "latitude", "sec_from_midnight", "status", "time"]
        for key, values in tqdm(data_dict.items()):
            for i, value in enumerate(values):
                df = pd.DataFrame(value, columns=columns)
                df.to_csv(os.path.join(self.csv_data_dir, f'driver_{key}_day_{i}.csv'), index=False)


    def process_df(self, df):
        df['time'] = pd.to_datetime(df["time"])
        df.sort_values(by="time")
        df['longitude'] = df['longitude'].round(3)
        df['latitude'] = df['latitude'].round(3)
        s1 = ((df['longitude'] <= self.max_long) & (df['longitude'] >= self.min_long))
        s2 = ((df['latitude'] <= self.max_lat) & (df['latitude'] >= self.min_lat))
        slice_ = s1 & s2
        df = df[slice_]
        df['long'] = df['longitude'].apply(lambda x: int(
            self.max_grid_long * ((x - self.min_long) / (self.max_long - self.min_long)))
                                           )
        df['lat'] = df['latitude'].apply(lambda x: int(
            self.max_grid_lat - self.max_grid_lat * ((x - self.min_lat) / (self.max_lat - self.min_lat)))
                                         )
        return df

    def draw_trajectory(self, df):
        color = (255, 255, 255)
        thickness = 2
        image = np.zeros((self.max_grid_lat, self.max_grid_long), np.uint8)
        points = list(df[['long', 'lat']].to_numpy())
        points = np.asarray(points, np.int32)
        image = cv2.polylines(image, [points], False, color, thickness)
        return image

    def get_feature(self, df):
        df1 = df[df["status"] == 1]
        df0 = df[df["status"] == 0]
        first_image = self.draw_trajectory(df1)
        second_image = self.draw_trajectory(df0)
        feature = np.stack([first_image, second_image], axis=0)

        return feature

    def create_feature(self):
        filenames = os.listdir(self.csv_data_dir)
        for filename in tqdm(filenames):
            df = pd.read_csv(os.path.join(self.csv_data_dir, filename))
            df = self.process_df(df)
            feature = self.get_feature(df=df)
            npy_filename = filename.replace("csv", 'npy')
            np.save(os.path.join(self.grid_data_dir, npy_filename), feature)


def main():
    data_prep = DataPrep()
    data_prep.create_csv()
    data_prep.create_feature()


if __name__ == '__main__':
    main()


# In[2]:


import random
import pandas as pd


def create_script(instance=20000):
    train_lists = []
    for i in range(instance):
        base = random.randint(0, 499)
        compare = random.randint(0, 499)
        day_base = random.randint(0, 4)
        day_compare = random.randint(0, 4)
        if base == compare:
            label = 1
        else:
            label = 0
        train_lists.append([base, compare, day_base, day_compare, label])
    idx = random.sample(range(0, instance), instance // 2)
    j = 0
    for id_ in idx:
        base = train_lists[id_][0]
        train_lists[id_][1] = base
        train_lists[id_][-1] = 1
        j += 1
    print(j)
    train_script_df = pd.DataFrame(train_lists, columns=['base', 'compare', 'base_day', 'comp_day', 'label'])
    train_script_df.to_csv("./training_script.csv", index=False)


if __name__ == '__main__':
    create_script(instance=5000)


# In[3]:


import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CustomDataloader(Dataset):
    def __init__(
            self,
            data_list,
            data_dir="./grid_data"
    ):
        self.data_dir = data_dir
        self.data_list = data_list
        random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        base, compare, base_day, comp_day, y = self.data_list[idx]
        x1_dir = os.path.join(self.data_dir, f'driver_{base}_day_{base_day}.npy')
        x2_dir = os.path.join(self.data_dir, f'driver_{compare}_day_{comp_day}.npy')
        x1, x2 = np.load(x1_dir) / 255, np.load(x2_dir) / 255
        x1, x2 = x1.astype('float32'), x2.astype('float32')
        y = np.float32(y)
        return x1, x2, y


class LoadData:
    def __init__(self, batch_size, data_list):
        self.batch_size = batch_size
        self.filenames = data_list

    def load_data(self):
        dataset = CustomDataloader(data_list=self.filenames)
        data_loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size
        )
        return data_loader


def main():
    data_list = pd.read_csv("training_script.csv").values.tolist()
    random.shuffle(data_list)
    train_split = int(0.7*len(data_list))
    val_split = int(0.8*len(data_list))
    train_data_list = data_list[:train_split]
    vad_data_list = data_list[train_split:val_split]
    test_data_list = data_list[val_split:]
    data_loader = LoadData(batch_size=16, data_list=train_data_list)
    for i, (x1, x2, y) in enumerate(data_loader.load_data()):
        print(y)
        print(f'{i}: X1 shape: {x1.shape} | X2 shape: {x2.shape}| y shape: {y.shape}')


if __name__ == '__main__':
    main()


# In[1]:


import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*48*25, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def main():
    siamese = SiameseNet()
    x = torch.randn((16, 2, 48, 25))
    x1, x2 = siamese(x, x)
    print(x1.shape, x2.shape)


if __name__ == '__main__':
    main()


# In[7]:


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


# In[9]:


import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from tqdm import tqdm
from data_loader import LoadData
from data_preparation import create_folder
# from model import SiameseNet


# matplotlib.rc('xtick', labelsize=20)
# matplotlib.rc('ytick', labelsize=20)


class Trainer:
    def __init__(
            self,
            batch_size=16,
            epochs=5,
            lr=1e-3
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_val_epoch = None
        self.best_val_loss = 1e8
        self.save_dir = "./saved_models"
        create_folder(self.save_dir)

    def train(self, train_=True):
        data_list = pd.read_csv("training_script.csv").values.tolist()
        random.shuffle(data_list)
        train_split = int(0.8 * len(data_list))
        train_data_list = data_list[:train_split]
        vad_data_list = data_list[train_split:]
        train_loader = LoadData(batch_size=self.batch_size, data_list=train_data_list)
        val_loader = LoadData(batch_size=self.batch_size, data_list=vad_data_list)

        model_name = "siamese_12.h5"
#         label_weights = [0.33, 0.67]
#         label_weights = torch.tensor(label_weights).float()
#         label_weights = label_weights.to(self.device)
        model = SiameseNet().to(device=self.device)
        criterion = ContrastiveLoss()
#         criterion = nn.BCELoss()
#         optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9,
                    momentum=0.5,
                    eps=1e-07,
                    centered=False)
        optimizer = optim.Adam(model.parameters(),lr = 0.0005)
#         lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)
        if train_:
            for epoch in range(self.epochs):
                model.train()
                train_loss = 0
                count = 0
                train_y = []
                pred_train_y = []
                progress_bar = tqdm(train_loader.load_data())
                progress_bar.set_description(f'Epoch: {epoch + 1}')
                for x1, x2, y in progress_bar:
                    x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
#                     y_pred = model(x1, x2)
                    y1, y2 = model(x1, x2)
#                     print(y_pred)
                    y = y.unsqueeze(1)
#                     print(y_pred.dtype)
#                     print(y.dtype)
#                     y_pred = (y_pred>0.5).float()
                    loss = criterion(y1, y2, y)
#                     print(loss)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    count += 1
                    progress_bar.set_postfix({"loss": round(train_loss / count, 4)})
                    train_y = train_y + list(y.detach().cpu().numpy())
#                     y_pred = y_pred.detach().cpu().numpy()
#                     pred_train_y = pred_train_y + list(y_pred)
#                 print('pred: ', pred_train_y)
#                 print('true: ', train_y)
#                 train_accuracy = my_accuracy(y_true=train_y, y_pred=pred_train_y)
#                 print(f'train accuracy: {train_accuracy}')
#                 lr_scheduler.step()
                optimizer.step()
                val_progress_bar = tqdm(val_loader.load_data())
                val_loss = 0
                count = 0
                val_y = []
                pred_val_y = []
                val_progress_bar.set_description(f'Validating: ')
                model.eval()
                for x1, x2, y in val_progress_bar:
                    x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                    y1, y2 = model(x1, x2)
#                     print(y_pred)
#                     y_pred = (y_pred>0.5).float()
                    y = y.unsqueeze(1)
                    loss = criterion(y1, y2, y)
                    val_loss += loss.item()
                    count += 1
                    val_progress_bar.set_postfix({"loss": round(val_loss / count, 4)})
                    val_y = val_y + list(y.detach().cpu().numpy())
#                     y_pred = y_pred.detach().cpu().numpy()
#                     pred_val_y = pred_val_y + list(y_pred)
                val_loss /= count
#                 accuracy = my_accuracy(y_true=val_y, y_pred=pred_val_y)
#                 print(f'Validation accuracy: {accuracy}')
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(self.save_dir, model_name))
                    print(f'Model saved at epoch {epoch + 1}')
                print("\n")
            print(f'\nBest model saved at epoch {self.best_val_epoch} with validation loss: {self.best_val_loss}')

batch_size = 128
epochs = 50
train = True

trainer = Trainer(
        batch_size=batch_size,
        epochs=epochs
    )
trainer.train(train_=train)


# In[6]:


import os
model = SiameseNet()
save_dir = "./saved_models"
model_name = "siamese_12.h5"
model.load_state_dict(torch.load(os.path.join(save_dir, model_name), map_location='cpu'))


# In[11]:


from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
data_list = pd.read_csv("training_script.csv").values.tolist()
np.random.shuffle(data_list)
random_split = int(0.96 * len(data_list))
train_data_list = data_list[:random_split]
val_data_list = data_list[random_split:]

val_loader = LoadData(batch_size=len(val_data_list), data_list=val_data_list)
val_progress_bar = tqdm(val_loader.load_data())
val_progress_bar.set_description(f'Validating: ')
# model.eval()
val_y = []
pred_val_y = []
y_pred = []
for x1, x2, y in val_progress_bar:
#     x1, x2, y = x1.to('cuda'), x2.to('cuda'), y.to('cuda')
    y1, y2 = model(x1, x2)
#                     print(y_pred)
#                     y_pred = (y_pred>0.5).float()
    y = y.unsqueeze(1)
    euclidean_distance = F.pairwise_distance(y1, y2)
    for i in range(len(euclidean_distance)):
        if euclidean_distance[i] < 0.4:
            y_pred.append(0)
        else: 
            y_pred.append(1)
#     print(euclidean_distance.item())
    print(classification_report(y, y_pred))
#                     y_pred = y_pred.detach().cpu().numpy()
#                     pred_val_y = pred_val_y + list(y_pred)
# val_loss /= count
#                 accuracy = my_accuracy(y_true=val_y, y_pred=pred_val_y)
#                 print(f'Validation accuracy: {accuracy}')


# In[ ]:




