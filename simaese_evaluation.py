#!/usr/bin/env python
# coding: utf-8

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


# In[2]:


import os
model = SiameseNet()
save_dir = "./saved_models"
model_name = "siamese_12.h5"
model.load_state_dict(torch.load(os.path.join(save_dir, model_name), map_location='cpu'))


# In[3]:


import torch
import pickle
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
np.random.seed(42)
pd.options.mode.chained_assignment = None  # default='warn'


random.seed(42)


class DataPrep:
    def __init__(self):
        self.min_lat = 22.502
        self.max_lat = 22.751
        self.min_long = 113.813
        self.max_long = 114.293
        self.max_grid_long = 48
        self.max_grid_lat = 25

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

    def create_feature(self, data_list):
        columns = ["longitude", "latitude", "sec_from_midnight", "status", "time"]
        df1 = self.process_df(pd.DataFrame(data_list[0], columns=columns))
        df2 = self.process_df(pd.DataFrame(data_list[1], columns=columns))
        x1 = self.get_feature(df1)
        x2 = self.get_feature(df2)
        return np.expand_dims(x1, axis=0), np.expand_dims(x2, axis=0)


with open("validate_set.pkl", "rb") as f:
    data_list = pickle.load(f)
f.close()
with open("validate_label.pkl", "rb") as f:
    label_list = pickle.load(f)
f.close()
data_prep = DataPrep()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = SiameseNet()
# model.load_state_dict(torch.load("./saved_models/siamese_10.h5"))
y_true, y_pred = [], []
for i, data in tqdm(enumerate(data_list)):
    x1, x2 = data_prep.create_feature(data)
    x1, x2 = torch.from_numpy(x1/255).float(), torch.from_numpy(x2/255).float()
#         x1, x2 = x1.to(device=device), x2.to(device=device)
    y1, y2 = model(x1, x2)
    euclidean_distance = F.pairwise_distance(y1, y2)
    if euclidean_distance < 0.4:
        y_pred.append(0)
    else: 
        y_pred.append(1)
#         y_ = torch.argmax(y_, dim=1)
#         y_pred += list(y_.detach().cpu().numpy())
    y_true.append(label_list[i])
print(classification_report(y_true=y_true, y_pred=y_pred))
print(confusion_matrix(y_true=y_true, y_pred=y_pred))



# evaluate()


# In[ ]:




