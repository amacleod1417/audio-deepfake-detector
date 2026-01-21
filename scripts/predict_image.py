import sys, os, torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

IMG = sys.argv[1]
CLASSES = ["fake","real"]  

tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

img = Image.open(IMG).convert("RGB")
x = tf(img).unsqueeze(0)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model_path = "runs/exp1/best_model.pt"
if not os.path.exists(model_path):
    model_path = "runs/exp1/model_state_dict.pt"

state = torch.load(model_path, map_location="cpu")
if "model_state" in state:
    state = state["model_state"]

model.load_state_dict(state)
model.eval()

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
    i = int(torch.argmax(probs))
    print(f"Pred: {CLASSES[i]} | probs={probs.tolist()}")
