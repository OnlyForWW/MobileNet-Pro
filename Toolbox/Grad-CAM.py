import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from MobileNetPro import *

model = MobileNetPro_100(num_classes=7)
model.load_state_dict(torch.load('best.pth'))

img = Image.open("7082.jpg").convert('RGB')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)

target_layers = [model.conv_layers[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

# targets = [ClassifierOutputTarget()]
img = np.array(img)
grayscale_cam = cam(input_tensor=img_tensor)
grayscale_cam = grayscale_cam[0, :]
print(type(img))
visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()