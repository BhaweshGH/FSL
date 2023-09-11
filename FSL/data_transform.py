import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
import pandas as pd
import os
from skimage import io
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SymbolDataset(Dataset):
    def __init__(self,csv_file, root_dir, transform= None):
        self.annotation= pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotation.iloc[index,0])
        image = io.imread(img_path)
        y_label = int(self.annotation.iloc[index,1])

        if self.transform:
            image= self.transform(image)

        return (image,y_label )


my_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor()
    ])


dataset = SymbolDataset(csv_file='Symbols_pre_test.csv', root_dir='./data', transform=my_transform)



num = 1

for _ in range(10):
    for img, label in dataset:
        save_image(img, './data/test_sym'+str(num)+'.png')
        # print(type(img))
        num += 1