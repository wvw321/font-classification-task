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
from matplotlib import pyplot as plt

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
        file_name: str = None,
        path: str = None,
        class_list: list = None,
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
        if class_list is not None:
            writer.writerow(class_list)
        else:
            writer.writerow(["avg"])
        for epoch_data in data:
            if isinstance(epoch_data, torch.Tensor):
                list_data = epoch_data.tolist()
                if isinstance(list_data, list):
                    writer.writerow(list_data)
                else:
                    writer.writerow([list_data])
            else:
                writer.writerow(epoch_data)


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


def avg_metric(data):
    return data.sum() / len(data)


def train_model(
        dataset_path: str,
        save_model: bool,
        save_model_path: str,
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

    kfold = KFold(n_splits=k_folds_num, shuffle=True)

    print('--------------------------------')
    print('--------------------------------')
    print(
        "Train param:\n"
        "    Device-{}\n"
        "    batch_size {},num_epochs {},learning_rate {},k_folds_num {},momentum {},weight_decay {}".format(
            device,
            batch_size,
            num_epochs,
            learning_rate,
            k_folds_num,
            momentum,
            weight_decay
        )
    )
    print('--------------------------------')
    print('--------------------------------')
    dataset = DatasetFonts(path=dataset_path, preprocess=preprocess, train=True)
    test_loader = data_loader(dataset_path=dataset_path, batch_size=batch_size, preprocess=preprocess, test=True)

    # metric
    metric_accuracy = torchmetrics.classification.Accuracy(task="multiclass", average="none", num_classes=10)
    metric_precision = torchmetrics.classification.Precision(task="multiclass", average="none", num_classes=10)
    metric_recall = torchmetrics.classification.Recall(task="multiclass", average="none", num_classes=10)
    metric_f1 = torchmetrics.classification.F1Score(task="multiclass", average="none", num_classes=10)

    def metric_update(
            predicted_,
            target_
    ):
        metric_accuracy.update(predicted_, target_)
        metric_precision(predicted_, target_)
        metric_recall(predicted_, target_)
        metric_f1(predicted_, target_)

    values_loss_train = []
    values_loss_val = []

    values_accuracy = []
    values_accuracy_avg = []
    values_precision = []
    values_precision_avg = []
    values_recall = []
    values_recall_avg = []
    values_f1 = []
    values_f1_avg = []

    metric_accuracy.to(device)
    metric_precision.to(device)
    metric_recall.to(device)
    metric_f1.to(device)

    # K-fold Cross Validation model evaluation

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        valloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        total_step = len(trainloader)
        for epoch in range(num_epochs):
            model.train()
            values_loss_epoch_train = []
            values_loss_epoch_val = []

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

                values_loss_epoch_train.append(loss.to("cpu"))

            avg_loss_epoch_train = sum(values_loss_epoch_train) / len(values_loss_epoch_train)
            print('Fold [{}/{}] Epoch [{}/{}]\n'
                  'Train: \n    Loss - {:.4f}'.format(fold + 1,
                                                      k_folds_num,
                                                      epoch + 1,
                                                      num_epochs,
                                                      avg_loss_epoch_train))
            values_loss_train.append(avg_loss_epoch_train)

            # Validation
            with torch.no_grad():

                model.eval()
                for images, labels in valloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    _, target = torch.max(labels.data, 1)

                    metric_update(predicted, target)
                    loss = criterion(outputs, labels)
                    values_loss_epoch_val.append(loss.to("cpu"))
                    del images, labels, outputs

                avg_loss_epoch_val = sum(values_loss_epoch_val) / len(values_loss_epoch_val)

                accuracy_epoch = avg_metric(metric_accuracy.compute())
                precision_epoch = avg_metric(metric_precision.compute())
                recall_epoch = avg_metric(metric_recall.compute())
                f1_epoch = avg_metric(metric_f1.compute())

                values_loss_val.append(avg_loss_epoch_val)
                values_precision.append(metric_accuracy.compute())
                values_recall.append(metric_precision.compute())
                values_f1.append(metric_f1.compute())
                values_accuracy.append(metric_accuracy.compute())

                values_precision_avg.append(precision_epoch)
                values_recall_avg.append(recall_epoch)
                values_f1_avg.append(f1_epoch)
                values_accuracy_avg.append(accuracy_epoch)

                print('Val:')
                print('    Loss - {:.4f} '.format(avg_loss_epoch_val.item()))
                print('    Acc - {:.1f} %'.format(accuracy_epoch.item() * 100))
                print('    Precision - {:.2f} '.format(precision_epoch.item()))
                print('    Recall- {:.2f} '.format(recall_epoch.item()))
                print('    F1- {:.2f} '.format(f1_epoch.item()))
                print('------------')

                metric_accuracy.reset()
                metric_precision.reset()
                metric_recall.reset()
                metric_f1.reset()

    class_list = DatasetFonts.get_class_list(dataset_path)

    logging_metrics(data=values_precision,
                    file_name="Precision",
                    class_list=class_list,
                    type_metric="val")

    logging_metrics(data=values_recall,
                    file_name="Recall",
                    class_list=class_list,
                    type_metric="val")

    logging_metrics(data=values_f1,
                    file_name="F1",
                    class_list=class_list,
                    type_metric="val")

    logging_metrics(data=values_precision_avg,
                    file_name="Precision_avg",
                    type_metric="val")

    logging_metrics(data=values_recall_avg,
                    file_name="Recall_avg",
                    type_metric="val")

    logging_metrics(data=values_f1_avg,
                    file_name="F1_avg",
                    type_metric="val")

    logging_metrics(data=values_loss_val,
                    file_name="Loss",
                    type_metric="val")

    logging_metrics(data=values_loss_train,
                    file_name="Loss",
                    type_metric="train")





    metric_precision.plot(values_precision)[0].savefig('metrics/val/precision.png')
    metric_accuracy.plot(values_accuracy)[0].savefig('metrics/val/accuracy.png')
    metric_recall.plot(values_recall)[0].savefig('metrics/val/recall.png')
    metric_f1.plot(values_f1)[0].savefig('metrics/val/values_f1.png')

    plt.figure()
    plt.plot(list(map(lambda x: x.item(), values_loss_train)))[0].figure.savefig('metrics/train/loss.png')
    plt.figure()
    plt.plot(list(map(lambda x: x.item(), values_loss_val)))[0].figure.savefig('metrics/val/loss.png')
    metric_precision.plot(values_precision_avg)[0].savefig('metrics/val/precision_avg.png')
    metric_accuracy.plot(values_accuracy_avg)[0].savefig('metrics/val/accuracy_avg.png')
    metric_recall.plot(values_recall_avg)[0].savefig('metrics/val/recall_avg.png')
    metric_f1.plot(values_f1_avg)[0].savefig('metrics/val/values_f1_avg.png')

    # Test model

    with torch.no_grad():
        metric_roc = torchmetrics.classification.MulticlassROC(num_classes=10, average=None)
        values_loss_epoch_test = []
        model.eval()

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, target = torch.max(labels.data, 1)
            metric_update(predicted, target)
            metric_roc.update(outputs, target)
            loss = criterion(outputs, labels)
            values_loss_epoch_val.append(loss.to("cpu"))

            del images, labels, outputs

        avg_loss_epoch_val = sum(values_loss_epoch_val) / len(values_loss_epoch_val)
        accuracy_epoch = avg_metric(metric_accuracy.compute())
        precision_epoch = avg_metric(metric_precision.compute())
        recall_epoch = avg_metric(metric_recall.compute())
        f1_epoch = avg_metric(metric_f1.compute())

        print('--------------------------------')
        print('--------------------------------')
        print('Test:')
        print('    Loss - {:.4f} '.format(avg_loss_epoch_val.item()))
        print('    Acc - {:.1f} %'.format(accuracy_epoch.item() * 100))
        print('    Precision - {:.2f} '.format(precision_epoch.item()))
        print('    Recall- {:.2f} '.format(recall_epoch.item()))
        print('    F1- {:.2f} '.format(f1_epoch.item()))
        print('--------------------------------')
        print('--------------------------------')

        all_test_metric = [[avg_loss_epoch_val.item(), accuracy_epoch.item() * 100,
                           precision_epoch.item(), recall_epoch.item(), f1_epoch.item()]]
        name_list = ["Loss", "Acc", "Precision", "Recall", "F1"]
        logging_metrics(data=all_test_metric,
                        file_name="all_test_metric",
                        class_list=name_list,
                        type_metric="test")

        metric_roc.plot(score=True)[0].savefig('metrics/test/roc.png')

    if save_model is True:
        if save_model_path is None:
            torch.save(model.state_dict(), "weights.pth")
            print("Model save as : " + os.getcwd() + "\weights.pth")
        else:
            torch.save(model.state_dict(), save_model_path + "\weights.pth")
            print("Model save as : " + save_model_path + "\weights.pth")


def parse_opt(
):
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset",
        help="dataset folder"
    )
    parser.add_argument(
        "--k_folds_num",
        type=int,
        default=5,
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
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
    parser.add_argument(
        "--save_model",
        type=bool,
        default=True,
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image"
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
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
