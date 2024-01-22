import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
import torchvision.transforms as T


class DatasetFonts(Dataset):

    def __init__(self, path, preprocess):
        self.path = path
        self.cl = {}
        count = 0
        for dir_name in os.listdir(path):
            self.cl[dir_name] = count
            count += 1
        self.num_cl = self.cl.__len__()

        self.img_cl_list = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith('.jpg'):
                    img_tensor = preprocess(Image.open(os.path.normpath(root + '/' + str(file))))
                    cl = self.cl[os.path.basename(root)]
                    self.img_cl_list.append([img_tensor, cl])

    def __len__(self):
        return self.img_list.__len__()

    def __getitem__(self, index):
        img, cl = self.img_cl_list[index]
        return img, self.class_to_tensor(cl)

    def class_to_tensor(self, cl: int) -> torch.Tensor:
        tensor = torch.tensor((), dtype=torch.float64)
        tensor = tensor.new_zeros((1, self.num_cl))
        tensor[0][cl] = 1
        return tensor


preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if __name__ == "__main__":
    dir_path = os.getcwd() + '\\dataset\\train'
    dataset_traine=DatasetFonts(path=dir_path,preprocess=preprocess)
    x=dataset_traine[0]
    print(x)


    # from torchvision.models import resnet18, ResNet18_Weights
    # from torchvision.io import read_image
    #
    # img = read_image("F:\\Projects\\font-classification-task\\dataset\\test\\Aguante-Regular\\0.jpg")
    # weights = ResNet18_Weights.DEFAULT
    # model = resnet18(weights=weights)
    #
    # model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
    # model.eval()
    # preprocess = weights.transforms()
    #
    # # Step 3: Apply inference preprocessing transforms
    # batch = preprocess(img).unsqueeze(0)
    #
    # # Step 4: Use the model and print the predicted category
    # prediction = model(batch).squeeze(0).softmax(0)
    # class_id = prediction.argmax().item()
    # print()
