import tabulate
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet18

from Train_model import DatasetFonts
from Train_model import preprocess_resnet18


def print_rezult(prediction: torch.Tensor, class_list: list):
    data = [['id', 'class', '%'], ]
    count = 0
    for x in prediction:
        x = round(x.item(), 3)*100
        data.append([count, class_list[count], str(x) ])
        count += 1

    results = tabulate.tabulate(data)
    print(results)


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load("weights.pth"))
    model.eval()

    preprocess = preprocess_resnet18
    img = Image.open("F:\\Projects\\font-classification-task\\dataset\\test\\Aguante-Regular\\0.jpg")
    # img = read_image("F:\\Projects\\font-classification-task\\dataset\\test\\Aguante-Regular\\0.jpg")
    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_list = DatasetFonts.get_class_list("F:\\Projects\\font-classification-task\\dataset")
    print_rezult(prediction, class_list)

    class_id = prediction.argmax().item()
    print("Rezlt - - "+class_list[class_id] +"Сonfidence "+str(round(prediction.max().item(),4)*100)+" %")

if __name__ == "__main__":
    run()

# print(class_id)
