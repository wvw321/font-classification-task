import tabulate
import torch
from PIL import Image
from torchvision.models import resnet18

from Train import DatasetFonts
from Train import preprocess_resnet18


def print_result(
        prediction: torch.Tensor,
        class_list: list
):
    data = [['id', 'class', '%'], ]
    count = 0
    for x in prediction:
        x = round(x.item()* 100, 1)
        data.append([count, class_list[count], str(x)])
        count += 1

    results = tabulate.tabulate(data)
    print(results)


def run(
        img_path: str,
        dataset_path: str,
        class_list: str,
        weights: str,
):
    if dataset_path is not None:
        class_list = DatasetFonts.get_class_list(dataset_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load(weights))
    model.to(device)
    model.eval()

    preprocess = preprocess_resnet18
    img = Image.open(img_path)
    batch = preprocess(img).unsqueeze(0).to(device)
    prediction = model(batch).squeeze(0).softmax(0)
    print_result(prediction, class_list)
    class_id = prediction.argmax().item()
    print("Result - - " + class_list[class_id] + "Confidence " + str(round(prediction.max().item(), 4) * 100) + " %")


# noinspection PyTypeChecker
def parse_opt():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_path",
        type=str,
        default='example/TanaUncialSP.jpg',
        help="path to image"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="model\weights.pth",
        help="path to image",
    )

    parser.add_argument(
        "--class_list",
        type=list[str],
        default=['Aguante-Regular', 'AlumniSansCollegiateOne-Regular', 'ambidexter_regular', 'ArefRuqaaInk-Regular',
                 'better-vcr-5.2', 'BrassMono-Regular', 'GaneshaType-Regular', 'GhastlyPanicCyr', 'Realest-Extended',
                 'TanaUncialSP'],
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite")

    return parser.parse_args()


def main(
        opt
):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
