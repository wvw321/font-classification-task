import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image

import tabulate
from Train_model import DatasetFonts


def print_rezult(prediction:torch.Tensor,class_list:list):
    data = [['id', 'class', '%'],]
    count=0
    for x in prediction:
        x=round(x.item(),2)
        data.append([count,class_list[count],str(x)+' %'])
        count+=1


    results = tabulate.tabulate(data)
    print(results)






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(num_classes=10)
model.load_state_dict(torch.load("weights.pth"))
model.eval()

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
img=Image.open("F:\\Projects\\font-classification-task\\dataset\\test\\Aguante-Regular\\0.jpg")
# img = read_image("F:\\Projects\\font-classification-task\\dataset\\test\\Aguante-Regular\\0.jpg")
# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_list=DatasetFonts.get_class_list("F:\\Projects\\font-classification-task\\dataset")
print_rezult(prediction,class_list)
class_id = prediction.argmax().item()




print(class_id)
