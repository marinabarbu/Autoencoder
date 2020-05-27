import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import clasa_autoencoder as CA
from image_dataset import ImageDataset
import pickle

dataset_test1 = ImageDataset(root_dir='./patches_img300x300_inainte_12x12', transform=transforms.ToTensor())
dataset_test1 = list(dataset_test1)

dataset_test2 = ImageDataset(root_dir='./patches_img300x300_dupa_12x12', transform=transforms.ToTensor())
dataset_test2 = list(dataset_test2)

test_loader1 = DataLoader(dataset_test1, batch_size=1, shuffle=False, drop_last=True)
test_loader2 = DataLoader(dataset_test2, batch_size=1, shuffle=False, drop_last=True)

model = CA.Autoencoder()
model.load_state_dict(torch.load("model_compresie"))
# model.eval()

F1 = []
F2 = []

with torch.no_grad():
    for data in test_loader1:
        features1 = model.codare(data) # torch tensor de dimensiune 32 X 64 X 7
        F1.append(features1.detach().numpy().flatten()) # o lista de ~ 90000/4 = 18.496  vectori numpy de dimensiune (14336, )
    for data in test_loader2:
        features2 = model.codare(data)
        F2.append(features2.detach().numpy().flatten())


with open("F1_12x12", 'wb') as f:
    pickle.dump(F1, f)

with open("F2_12x12", 'wb') as f:
    pickle.dump(F2, f)
