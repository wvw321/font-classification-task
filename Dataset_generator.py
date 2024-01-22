# The generators use the same arguments as the CLI, only as parameters
import os
from pathlib import Path
from RandomWordGenerator import RandomWord
from trdg.generators import GeneratorFromStrings


def generate_data(count: int, max_word_size: int = 5, fonts: list = None, file_path: str = None):
    if fonts is None:
        fonts = []

    # Creating a random word object
    rw = RandomWord(max_word_size=max_word_size,
                    constant_word_size=False,
                    include_digits=True,
                    special_chars=r"@_!#$%^&*()<>?/\|}{~:",
                    include_special_chars=False)

    x = [rw.generate() for _ in range(count)]

    generator = GeneratorFromStrings(strings=x, count=count,
                                     fonts=fonts)

    img_name = 0
    for img, text in generator:
        if img is not None:
            img.save(file_path + "/" + str(img_name) + ".jpg")
            img_name += 1
        else:
            print("ошибка генерации")


def get_fronts_path(path: str) -> list:
    dir_path = os.path.normpath(os.getcwd() + '\\' + path)
    path_dict = {}
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.otf') or file.endswith('.ttf'):
                print(os.path.normpath(root + '/' + str(file)))
                path_dict[Path(file).stem] = os.path.normpath(root + '/' + str(file))
    return path_dict


def generate_dataset(count: int, fonts_path: str, file_path: str = None, ):
    def d(path_dict, folder):
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

    d(path_dict, test_folder)
    d(path_dict, train_folder)


if __name__ == "__main__":
    # generate_data(count=5 ,file_path="data")
    generate_dataset(fonts_path="fonts", count=100)
