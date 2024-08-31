from flask import Flask, request, jsonify
import torch 
from PIL import Image
from torchvision import transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
sys.path.append(r'D:\micro-doppler based target classification\Micro-Doppler-Based-Target-Classification-')
from ml_model.src.model.model_vgg import initialize_vgg19
from ml_model.src.model.model_vgg import CustomVGG

class CustomVGG(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomVGG, self).__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(base_model.classifier[0].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def initialize_vgg19(num_classes):
    vgg16_base = models.vgg16(pretrained=True)
    for param in vgg16_base.parameters():
        param.requires_grad = False
    return CustomVGG(vgg16_base, num_classes)


# load the ml model
num_classes = 6
model = initialize_vgg19(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(r'best_model_CustomVGG.pt'))
model.eval()

def predict(image_tensor):
    image_tensor = image_tensor.to(device)  # Move input data to GPU
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

app = Flask(__name__)

# define a transform to preprocess the input image 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict_route():
    # get the file from the POST request
    file = request.files['image']

    # convert the file to an image 
    image = Image.open(file.stream).convert('RGB')

    # preprocess the image 
    image = transform(image).unsqueeze(0)

    prediction = predict(image)

    # return the prediction as a json response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)