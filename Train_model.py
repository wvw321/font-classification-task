import csv
import os


import numpy as np
import torch
import torchmetrics
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models import resnet18
from sklearn.model_selection import KFold


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
        path = path + '//train'
        class_list = []
        for dir_name in os.listdir(path):
            class_list.append(dir_name)
        return class_list


def data_loader(data_dir: str,
                preprocess,
                batch_size: int,
                random_seed: int = 42,
                valid_size: float = 0.1,
                shuffle: bool = True,
                test: bool = False,
                ):
    if test:
        dataset = DatasetFonts(path=data_dir,
                               preprocess=preprocess,
                               train=False
                               )

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle
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


def logging_metrics(data, file_name: str = None, path: str = None, field: list = None):
    if file_name is None:
        file_name = 'data.csv'
    else:
        file_name = file_name + ".csv"
    if path is not None:
        file_name = path + file_name

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        if field is not None:
            writer.writerow(field)

        for epoxdata in data:
            listdata = epoxdata.tolist()
            writer.writerow(listdata)


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


    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(num_classes=10).to(device)
    num_epochs = 15
    batch_size = 64
    learning_rate = 0.0005

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)



    # train_loader = data_loader(data_dir='dataset',
    #                            batch_size=batch_size,
    #                            preprocess=preprocess,
    #                            kfold=True)

    dataset = DatasetFonts(path='dataset',
                           preprocess=preprocess,
                           train=True
                           )

    test_loader = data_loader(data_dir='dataset',
                              batch_size=64,
                              preprocess=preprocess,
                              test=True)

    # metric
    metric_Accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
    metric_Precision = torchmetrics.classification.Precision(task="multiclass", average=None, num_classes=10)
    metric_Recall = torchmetrics.classification.Recall(task="multiclass", average=None, num_classes=10)
    metric_F1 = torchmetrics.classification.F1Score(task="multiclass", average=None, num_classes=10)

    values_Accuracy = []
    values_Precision = []
    values_Recall = []
    values_F1 = []
    values_loss = []

    metric_Accuracy.to(device)
    metric_Precision.to(device)
    metric_Recall.to(device)
    metric_F1.to(device)

    k_folds_num = 5

    kfold = KFold(n_splits=k_folds_num, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  sampler=train_subsampler)
        validloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  sampler=test_subsampler)

        total_step = len(trainloader)
        for epoch in range(num_epochs):
            model.train()
            # Train
            for i, (images, labels) in enumerate(trainloader):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                # writer.add_scalar("Loss/train", loss.item(), epoch)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

                values_loss.append(loss)

            # Validation
            with torch.no_grad():
                correct = 0
                total = 0
                model.eval()
                for images, labels in validloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    _, target = torch.max(labels.data, 1)
                    correct += (predicted == target).sum().item()
                    metric_Accuracy.update(predicted, target)
                    metric_Precision.update(predicted, target)
                    metric_Recall.update(predicted, target)
                    metric_F1.update(predicted, target)
                    del images, labels, outputs

                print('Accuracy of the network on the {} validation images: {} %'.format(100, 100 * correct / total))
                Accuracy_epoch = round(metric_Accuracy.compute().item(), 2)
                Precision_epoch = metric_Precision.compute()
                Recall_epoch = metric_Recall.compute()
                F1_epoch = metric_F1.compute()
                values_Precision.append(Precision_epoch)
                values_Recall.append(Recall_epoch)
                values_F1.append(F1_epoch)
                # print(f"Accuracy on all data: {Accuracy_epoch}")
                # print(f"Precision on all data: {pr}")
                # print(f"Recal on all data: {re}")
                metric_Accuracy.reset()
                metric_Precision.reset()
                metric_Recall.reset()
                metric_F1.reset()

    metric_Precision.plot(values_Precision)
    metric_Recall.plot(values_Recall)
    class_list = DatasetFonts.get_class_list("F:\\Projects\\font-classification-task\\dataset")
    logging_metrics(data=values_Precision,
                    file_name="Precision",
                    field=class_list)

    logging_metrics(data=values_Recall,
                    file_name="Recall",
                    field=class_list)

    logging_metrics(data=values_F1,
                    file_name="F1",
                    field=class_list)

    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, target = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == target).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} test images: {} %'.format(1000, 100 * correct / total))

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
