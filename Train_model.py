import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
import torchvision.transforms as T
from torch.utils.data.sampler import SubsetRandomSampler


class DatasetFonts(Dataset):

    def __init__(self, path: str, preprocess: T.Compose, train: bool = True):

        if train is True:
            self.path = path + '//train'
        else:
            self.path = path + '//test'

        self.cl = {}
        count = 0
        for dir_name in os.listdir(self.path):
            self.cl[dir_name] = count
            count += 1
        self.num_cl = self.cl.__len__()

        self.img_cl_list = []
        for root, dirs, files in os.walk(os.path.abspath(self.path)):
            for file in files:
                if file.endswith('.jpg'):
                    img_tensor = preprocess(Image.open(os.path.normpath(root + '/' + str(file))))
                    cl = self.cl[os.path.basename(root)]
                    self.img_cl_list.append([img_tensor, cl])

    def __len__(self):
        return self.img_cl_list.__len__()

    def __getitem__(self, index):
        img, cl = self.img_cl_list[index]
        return img, self.class_to_tensor(cl)

    def class_to_tensor(self, cl: int) -> torch.Tensor:
        tensor = torch.tensor((), dtype=torch.float64)
        tensor = tensor.new_zeros(self.num_cl)
        tensor[cl] = 1
        return tensor

    @staticmethod
    def get_class_list(path):
        path=path + '//train'
        class_list=[]
        for dir_name in os.listdir(path):
            class_list.append(dir_name)
        return class_list




preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def data_loader(data_dir,
                preprocess,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
    if test:
        dataset = DatasetFonts(path=data_dir, preprocess=preprocess, train=False)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = DatasetFonts(path=data_dir, preprocess=preprocess, train=True)

    valid_dataset = train_dataset

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


if __name__ == "__main__":
    # dir_path = os.getcwd() + '\\dataset\\train'
    # dataset_traine=DatasetFonts(path=dir_path,preprocess=preprocess)
    # x=dataset_traine[0]
    # print(x)

    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision.io import read_image

    # preprocess = ResNet18_Weights.transforms

    img = read_image("F:\\Projects\\font-classification-task\\dataset\\test\\Aguante-Regular\\0.jpg")
    # weights = ResNet18_Weights.DEFAULT
    # model = resnet18(weights=weights, num_classes=10)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(num_classes=10).to(device)
    num_epochs = 50
    batch_size = 16
    learning_rate = 0.0005

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.8)

    train_loader, valid_loader = data_loader(data_dir='dataset',
                                             batch_size=64,
                                             preprocess=preprocess)

    test_loader = data_loader(data_dir='dataset',
                              batch_size=64,
                              preprocess=preprocess,
                              test=True)

    # Train the model
    total_step = len(train_loader)

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            model.eval()
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                _, predicted_l = torch.max(labels.data, 1)
                correct += (predicted == predicted_l).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_l = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == predicted_l).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))

    torch.save(model.state_dict(), "weights.pth")

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
