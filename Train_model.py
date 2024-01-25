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

    def __init__(
            self,
            path: str,
            preprocess: T.Compose,
            train: bool = True
    ):

        if train is True:
            self.path = path + '//train'
        else:
            self.path = path + '//test'

        self.cl = {}
        count = 0
        for dir_name in os.listdir(os.path.abspath(self.path)):
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


def data_loader(
        dataset_path: str,
        preprocess,
        batch_size: int,
        random_seed: int = 42,
        valid_size: float = 0.1,
        shuffle: bool = True,
        test: bool = False,
):
    if test:
        dataset = DatasetFonts(path=dataset_path,
                               preprocess=preprocess,
                               train=False
                               )

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle
                                                  )

        return data_loader

    # load the dataset
    train_dataset = DatasetFonts(path=dataset_path, preprocess=preprocess, train=True)

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

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               sampler=valid_sampler)

    return (train_loader, valid_loader)


def logging_metrics(
        data,
        file_name:
        str = None,
        path: str = None,
        field: list = None,
        type_metric: str = None,

):
    if file_name is None:
        file_name = 'data.csv'
    else:
        file_name = file_name + ".csv"

    if path is None:
        if type_metric is None:
            if not os.path.isdir("metrics"):
                os.mkdir("metrics")

            file_name = "metrics/" + file_name
        else:
            if not os.path.isdir("metrics"):
                os.mkdir("metrics")
            if not os.path.isdir("metrics/" + type_metric):
                os.mkdir("metrics/" + type_metric)
            file_name = "metrics/" + type_metric + "/" + file_name
    else:
        if type_metric is None:
            if not os.path.isdir(path + "/metrics"):
                os.mkdir(path + "/metrics")
            file_name = path + "/metrics/" + file_name
        else:
            if not os.path.isdir(path + "/metrics"):
                os.mkdir(path + "/metrics")
            if not os.path.isdir(path + "/metrics/" + type_metric):
                os.mkdir(path + "/metrics/" + type_metric)
                file_name = path + "/metrics/" + type_metric + "/" + file_name

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        if field is not None:
            writer.writerow(field)
        else:
            writer.writerow(["avg"])
        for epoxdata in data:
            if isinstance(epoxdata, torch.Tensor):
                listdata = epoxdata.tolist()
                if isinstance(listdata, list):
                    writer.writerow(listdata)
                else:
                    writer.writerow([listdata])
            else:
                print([epoxdata])
                writer.writerow([epoxdata])


def plot_metric():
    pass


preprocess_resnet18 = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def train_model(
        dataset_path: str,
        num_epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 0.0005,
        k_folds_num: int = 4,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        preprocess=None,
        device=None,
        optimizer=None,
        criterion=None,
        model=None,

):
    # Device configuration
    if preprocess is None:
        preprocess = preprocess_resnet18
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    if model is None:
        model = resnet18(num_classes=10).to(device)
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    print('--------------------------------')
    print('--------------------------------')
    print("Train param:\n"
          "Device-{}\n"
          "batch_size {},num_epochs {},learning_rate {},k_folds_num {},momentum {},weight_decay {}".format(device,
                                                                                                           batch_size,
                                                                                                           num_epochs,
                                                                                                           learning_rate,
                                                                                                           k_folds_num,
                                                                                                           momentum,
                                                                                                           weight_decay
                                                                                                           )
          )
    print('--------------------------------')
    dataset = DatasetFonts(path=dataset_path, preprocess=preprocess, train=True)
    test_loader = data_loader(dataset_path=dataset_path, batch_size=batch_size, preprocess=preprocess, test=True)

    # metric
    metric_Accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
    metric_Precision = torchmetrics.classification.Precision(task="multiclass", average=None, num_classes=10)
    metric_Recall = torchmetrics.classification.Recall(task="multiclass", average=None, num_classes=10)
    metric_F1 = torchmetrics.classification.F1Score(task="multiclass", average=None, num_classes=10)

    values_Accuracy = []
    values_Precision = []
    values_Recall = []
    values_F1 = []
    values_Loss = []

    metric_Accuracy.to(device)
    metric_Precision.to(device)
    metric_Recall.to(device)
    metric_F1.to(device)

    kfold = KFold(n_splits=k_folds_num, shuffle=True)

    # Start print

    # K-fold Cross Validation model evaluation

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print('--------------------------------')
        print(f'FOLD {fold}')
        print('------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        valloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        total_step = len(trainloader)
        for epoch in range(num_epochs):
            model.train()
            values_loss_epoch = []
            # Train
            for i, (images, labels) in enumerate(trainloader):
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

                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

                values_loss_epoch.append(loss.to("cpu"))

            avg_loss_epoch = sum(values_loss_epoch) / len(values_loss_epoch)
            print('Epoch [{}/{}]\nTrain: Loss - {:.4f}'.format(epoch + 1,
                                                               num_epochs,
                                                               avg_loss_epoch))
            values_Loss.append(avg_loss_epoch)

            # Validation
            with torch.no_grad():
                correct = 0
                total = 0
                model.eval()
                for images, labels in valloader:
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

                Accuracy_epoch = round(metric_Accuracy.compute().item(), 2) * 100
                Precision_epoch = metric_Precision.compute()
                Recall_epoch = metric_Recall.compute()
                F1_epoch = metric_F1.compute()
                #
                values_Precision.append(Precision_epoch)
                values_Recall.append(Recall_epoch)
                values_F1.append(F1_epoch)
                values_Accuracy.append(Accuracy_epoch)
                print('Val: Acc - {} %'.format(Accuracy_epoch))
                print('------------')
                # print(f"Accuracy on all data: {Accuracy_epoch}")
                # print(f"Precision on all data: {pr}")
                # print(f"Recal on all data: {re}")
                metric_Accuracy.reset()
                metric_Precision.reset()
                metric_Recall.reset()
                metric_F1.reset()

    # Test model
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
        print('--------------------------------')
        print('--------------------------------')
        print('Accuracy of the network on the {} test images: {} %'.format(1000, 100 * correct / total))

    metric_Precision.plot(values_Precision)
    metric_Recall.plot(values_Recall)
    class_list = DatasetFonts.get_class_list(dataset_path)

    logging_metrics(data=values_Precision,
                    file_name="Precision",
                    field=class_list,
                    type_metric="val")

    logging_metrics(data=values_Recall,
                    file_name="Recall",
                    field=class_list,
                    type_metric="val")

    logging_metrics(data=values_F1,
                    file_name="F1",
                    field=class_list,
                    type_metric="val")

    logging_metrics(data=values_Loss,
                    file_name="Loss",
                    type_metric="train")

    torch.save(model.state_dict(), "weights.pth")


def parse_opt():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset",
        help="fonts folder"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="the path to the directory where the dataset will be generated"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="the number of instances of one class"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="Define skewing angle of the generated text. In positive degrees"
    )
    parser.add_argument(
        "--k_folds_num",
        type=int,
        default=5,
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Define the text's color, should be either a single hex color or a range in the ?,? format.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image"
    )

    return parser.parse_args()


def main(
        opt
):
    train_model(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
