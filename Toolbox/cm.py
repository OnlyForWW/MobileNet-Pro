import itertools
import matplotlib.pyplot as plt
import torchmetrics
import numpy as np
import torchvision
from MobileNetPro import *
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Normalize

test_dir = './datas/test'

test_dataset = datasets.ImageFolder(test_dir, transform=torchvision.transforms.Compose([Resize([224, 224]),
                                                                                        ToTensor(),
                                                                                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

cm = torchmetrics.ConfusionMatrix(num_classes=7, task="multiclass")
net = MobileNetPro_100(num_classes=7)
net.load_state_dict(torch.load('best.pth'))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cm.to(device)
net = net.to(device)
net.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        cm.update(preds=predicted, target=labels)

confmat = cm.compute()
print(confmat)

confmat = confmat.cpu()
confmat = confmat.numpy()

classes = ["label0", "label1", "label2", "label3", "label4", "label5", "label6"]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confmat, cmap=plt.cm.Blues)

ax.set_xlabel('Predicted Labels', fontsize=16)
ax.set_ylabel('True Labels', fontsize=16)

tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=12, rotation=45, ha='right')
ax.set_yticklabels(classes, fontsize=12)

thresh = confmat.max() / 2.
for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
    ax.text(j, i, format(confmat[i, j], 'd'),
             ha="center", va="center",
             color="white" if confmat[i, j] > thresh else "black")

plt.show()
plt.savefig("./CM_MobileNet-Pro.png")