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
    vgg19_base = models.vgg19(pretrained=True)
    for param in vgg19_base.parameters():
        param.requires_grad = False
    return CustomVGG(vgg19_base, num_classes)


# load the ml model
num_classes = 6
model = initialize_vgg19(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
try:
    model.load_state_dict(torch.load(r"D:\micro-doppler based target classification\Micro-Doppler-Based-Target-Classification-\Backend\best_model_CustomVGG.pt", weights_only=True))
except RuntimeError as e:
    print("Error loading model state dict:", e)

def predict(image_tensor, class_names):
    image_tensor = image_tensor.to(device)  
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item()


app = Flask(__name__)

@app.route('/')
def index():
    return " "

# define a transform to preprocess the input image 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict_route():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image = transform(image).unsqueeze(0)
    class_names = ['3 long blade rotor', '3 short blade rotor', 'Bird', 'Bird + mini-helicopter', 'Drone', 'RC Plane']
    prediction, confidence = predict(image, class_names)

    return jsonify({'prediction': prediction, 'confidence': confidence})


if __name__ == '__main__':
    app.run(debug=True)