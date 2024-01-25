import argparse
import os
import warnings
from pathlib import Path

from RandomWordGenerator import RandomWord
from trdg.generators import GeneratorFromStrings


def generate_data(
        max_word_size: int = 5,
        fonts: list = None,
        file_path: str = None,
        **kwargs
):
    if fonts is None:
        fonts = []

    # Creating a random word object
    rw = RandomWord(max_word_size=max_word_size,
                    constant_word_size=False,
                    include_digits=True,
                    special_chars=r"@_!#$%^&*()<>?/\|}{~:",
                    include_special_chars=False)

    x = [rw.generate() for _ in range(kwargs["count"])]

    generator = GeneratorFromStrings(strings=x,
                                     fonts=fonts,
                                     **kwargs
                                     )

    img_name = 0
    for img, text in generator:
        if img is not None:
            img.save(file_path + "/" + str(img_name) + ".jpg")
            img_name += 1
        else:
            warnings.warn("Еrror generating an image in the library: trdg")


def get_fronts_path(
        path: str
) -> dict:
    dir_path = os.path.abspath(path)
    path_dict = {}
    print("Found fonts")
    print('--------------------------------')
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.otf') or file.endswith('.ttf'):
                print(os.path.normpath(root + '/' + str(file)))
                path_dict[Path(file).stem] = os.path.normpath(root + '/' + str(file))
    print('--------------------------------')
    return path_dict


def generate_dataset(
        fonts_path: str,
        file_path: str = None,
        **kwargs
):
    def generate_folder(
            path_dict: dict,
            folder: str
    ):
        for key in path_dict:
            font = [path_dict[key]]
            path = folder + "/" + key
            if not os.path.isdir(path):
                os.mkdir(path)
            generate_data(file_path=path, fonts=font, **kwargs)

    if file_path is None:
        file_path = os.getcwd()

    dataset_folder = file_path + "/" + "dataset"
    test_folder = dataset_folder + "/" + "test"
    train_folder = dataset_folder + "/" + "train"

    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.isdir(test_folder):
        os.mkdir(test_folder)
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)
    path_dict = get_fronts_path(fonts_path)

    generate_folder(path_dict, test_folder)
    generate_folder(path_dict, train_folder)


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fonts_path",
        type=str,
        default="fonts",
        help="fonts folder"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="the path to the directory where the dataset will be generated"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="the number of instances of one class"
    )
    parser.add_argument(
        "--skewing_angle",
        type=int,
        default=5,
        help="Define skewing angle of the generated text. In positive degrees"
    )
    parser.add_argument(
        "--random_skew",
        type=bool,
        default=True,
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite")
    parser.add_argument(
        "--text_color",
        type=str,
        nargs="?",
        default="#000000,#888888",
        help="Define the text's color, should be either a single hex color or a range in the ?,? format.",
    )
    parser.add_argument(
        "--background_type",
        type=int,
        nargs="?",
        default=1,
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image"
    )
    return parser.parse_args()


def main(
        opt
):
    generate_dataset(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
