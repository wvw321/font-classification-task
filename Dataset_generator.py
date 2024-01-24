# The generators use the same arguments as the CLI, only as parameters
import os
from pathlib import Path
import warnings
from RandomWordGenerator import RandomWord
from trdg.generators import GeneratorFromStrings
import argparse


def generate_data(count: int,
                  max_word_size: int = 5,
                  fonts: list = None,
                  file_path: str = None):
    if fonts is None:
        fonts = []

    # Creating a random word object
    rw = RandomWord(max_word_size=max_word_size,
                    constant_word_size=False,
                    include_digits=True,
                    special_chars=r"@_!#$%^&*()<>?/\|}{~:",
                    include_special_chars=False)

    x = [rw.generate() for _ in range(count)]

    generator = GeneratorFromStrings(strings=x,
                                     count=count,
                                     fonts=fonts,
                                     skewing_angle=5,
                                     random_skew=True, )

    img_name = 0
    for img, text in generator:
        if img is not None:
            img.save(file_path + "/" + str(img_name) + ".jpg")
            img_name += 1
        else:
            warnings.warn("Еrror generating an image in the library: trdg")


def get_fronts_path(path: str) -> dict:
    dir_path = os.path.normpath(os.getcwd() + '\\' + path)

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


def generate_dataset(count: int,
                     fonts_path: str,
                     file_path: str = None
                     ):
    def generate_folder(path_dict, folder):
        for key in path_dict:
            font = [path_dict[key]]
            path = folder + "/" + key
            if not os.path.isdir(path):
                os.mkdir(path)
            generate_data(count=count, file_path=path, fonts=font)

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
    parser.add_argument("--fonts_path", type=str, default="fonts", help="fonts folder")
    parser.add_argument("--file_path", type=str, default=None, help="file_path folder")
    parser.add_argument("--count", type=int, default=50, help="file_path folder")
    # parser.add_argument("--source", type=str, required=True, help="video file path")
    # parser.add_argument("--view-img", action="store_true", help="show results")
    # parser.add_argument("--save-img", action="store_true", help="save results")
    # parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    return parser.parse_args()


def main(opt):
    generate_dataset(**vars(opt))


# if __name__ == "__main__":
#
#     # generate_data(count=5 ,file_path="data")
#     generate_dataset(fonts_path="fonts", count=100)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
