# Тестовое задание: генерация и распознование шрифтов 
## Установка зависимостей 
Версия - Python 3.10

Установка  [Torch](https://pytorch.org/get-started/locally/) 
версии cuda. Нужно предварительно установить ,если нужен запуск и обучение на gpu.
```
pip install -r requirements.txt
```

## Генератор датасета шрифтов 
[Generate_dataset.py](https://github.com/wvw321/font-classification-task/blob/main/Generate_dataset.py)

Основная чать генератора использует библиотеку [trdg](https://github.com/Belval/TextRecognitionDataGenerator/tree/master) 

### Структура папки с шрифтами
```
fonts/
|
|-- font1/
|   |-- font1.otf
|   |
|-- font2/
|   |-- font2.ttf
|-- .../
|   |   |-- ...

```

 ### Структура  генерируемого датасета
```
Dataset/
|
|-- train/
|   |-- class1/
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
|   |
|   |-- class2/
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
|   |
|   |-- ...
|
|-- test/
|   |-- class1/
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
|   |
|   |-- class2/
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
|   |
|   |-- ...
```

### Парамеры
- `--fonts_path` -путь для папки со шрифтами 
- `--file_path` -путь куда будет сгенерирован датасет
- `--count` -количество изображений генерируемое для одного класса
- `--skewing_angle` -угол наклона сгенерированного текста. В положительных градусах
- `--random_skew` -при установке угол наклона будет рандомизирован между значением, заданным с помощью -k, и его противоположным значением
- `--text_color` -определите цвет текста, он должен быть либо одним шестнадцатеричным цветом, либо диапазоном в формате ?,?.
- `--background_type` -определите, какой фон использовать. 0: Гауссовский шум, 1: Обычный белый, 2: 
### Пример вызова 
По умолчанию датасет генерируеться в  коталог проекта  ,
и ищет папку  fonts там же. 

```
Generate_dataset.py
```
Опция выбора количества одного изображений класса.
```
Generate_dataset.py --count 150
```
## Модуль обучения 
[Train_model.py](https://github.com/wvw321/font-classification-task/blob/main/Train.py)

Генеригует папку с даннми метрик и графиками
```
metric/
|
|-- test/
|   |-- all_testmetric.csv
|   |-- roc.png
| 
|-- train/
|   |-- loss.csv
|   |-- loss.png
| 
|-- val/
|   |-- Loss.csv
|   |-- Accuracy.csv
|   |-- Precision.csv
|   |-- Recall.csv
|   |-- F1.csv
|   |-- Accuracy_avg.csv
|   |-- Precision_avg.csv
|   |-- Recall_avg.csv
|   |-- F1_avg.csv
|   |-- loss.png
|   |-- accuracy.png
|   |-- precision.png
|   |-- recall.png
|   |-- f1.png
|   |-- accuracy_avg.png
|   |-- precision_avg.png
|   |-- recall_avg.png
|   |-- f1_avg.png
```
### Парамеры

- `--dataset_path` -путь до датасета 
- `--k_folds_num` -количество на которое будет разделена обучающая выборка для кросс валидации  
- `--num_epochs` -количество эпох на одну выборку
- `--batch_size` - количество изображений в батче 
- `--learning_rate` -темп обучения 
- `--momentum` -   момент ипульса стохастического градиентного спуска(оптимизатор SGD по умолчанию)
- `--weight_decay` - L_2 регуляризация
- `--save_model` -  флаг сохронять ли модель
- `--save_model_path` - путь сохранения модели ( по умолчанию дерриктория проекта)
### Пример вызова 
Запускает тернеровку со стандартными парпаметрами и ищет датасет в каталоге проекта.
```
Train.py  
```
Пример собственных папраметров
```
Train.py  --num_epochs 5 --k_folds_num 4 --dataset_path dataset --batch_size 16
```
## Модуль распознования 
[Predict.py](https://github.com/wvw321/font-classification-task/blob/main/Predict.py)
### Парамеры
- `--img_path` -путь до изображения 
- `--weights` - путь до фала весов (weights.pth)
- `--class_list` - названия классов 
- `--dataset_path` - названия классов можно получить указав путь до датасета
  ### Пример вызова 
По умолчанию инициирует веса model\trained_weights.pth  , и берет пример  example/TanaUncialSP.jpg

```
Predict.py  --img_path front.jpg
```

```
Predict.py  --img_path front.jpg
```

  
