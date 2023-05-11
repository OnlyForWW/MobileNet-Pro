import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Normalize
from tqdm import tqdm
from MobileNetPro import *
import timm.utils

test_dir = './datas/test'


test_dataset = datasets.ImageFolder(test_dir, transform=torchvision.transforms.Compose([Resize([224, 224]),
                                                                                        ToTensor(),
                                                                                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

batch_size = 100
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"测试集长度为：{len(test_dataset)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = MobileNetPro_100(num_classes=7)
net = net.to(device=device)

ema_model = timm.utils.ModelEmaV2(model=net, decay=0.9, device=device)
ema_model.load_state_dict(torch.load("./best_ema.pth"))

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device=device)

net.eval()
eval_loss = 0
eval_acc = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc='测试'):
        imgs, targets = batch
        imgs = imgs.to(device=device)
        targets = targets.to(device=device)
        output = ema_model.module(imgs)
        Loss = loss_fn(output, targets)
        _, pred = output.max(1)

        num_correct = (pred == targets).sum().item()
        eval_loss += Loss
        eval_acc += num_correct
    print(eval_acc)
    eval_losses = eval_loss / (len(test_dataset))
    eval_acc = eval_acc / (len(test_dataset))
    print(f"测试集上的Loss: {eval_losses}")
    print(f"测试集上的正确率: {eval_acc}")