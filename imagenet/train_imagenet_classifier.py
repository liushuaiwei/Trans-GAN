import os
import math
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from imagenet.dataset import ImageNetDataset
from imagenet.classifier import shufflenet_v2_x1_0


parser = argparse.ArgumentParser(description="Train for imagenet")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--saved_path', type=str, default='train_DICnn_Mini_ImageNet_no_normalize/')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# Load dataset
print('Loading dataset ...\n')
dataset_path = '/remote-home/cs_igps_lsw/AdversarialProjects/DICNN_mini_imagenet/data/no_normalize/'

# 准备训练数据集
train_data = np.load(dataset_path + 'train_DP_76925/xs_mini_imagenet.npy')
train_label = np.load(dataset_path + 'train_DP_76925/ys_mini_imagenet.npy')
train_transform = transforms.Compose([transforms.Resize(96),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
train_dataset = ImageNetDataset(train_data, train_label, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

# 准备测试数据集
test_data = np.load(dataset_path + 'test_data/None/xs_mini_imagenet.npy')
test_label = np.load(dataset_path + 'test_data/None/ys_mini_imagenet.npy')
test_transform = transforms.Compose([transforms.Resize(96),
                                     transforms.ToTensor()])
test_dataset = ImageNetDataset(test_data, test_label, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)

# 初始化分类模型
model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)

pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
# Scheduler https://arxiv.org/pdf/1812.01187.pdf
lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

for epoch in range(args.epochs):
    # train
    mean_loss = train_one_epoch(model=model,
                                optimizer=optimizer,
                                data_loader=train_loader,
                                device=device,
                                epoch=epoch,
                                warmup=True)

    scheduler.step()

    # validate
    acc = evaluate(model=model,
                   data_loader=val_loader,
                   device=device)

    print("[epoch {}] accuracy: {}".format(epoch, round(acc, 5)))
    # tags = ["loss", "accuracy", "learning_rate"]
    # tb_writer.add_scalar(tags[0], mean_loss, epoch)
    # tb_writer.add_scalar(tags[1], acc, epoch)
    # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
    print("current best_acc:", best_acc)
    if acc > best_acc:
        best_acc = acc
        print("new best_acc:", best_acc)
        torch.save(model.state_dict(), opt.save_path + "best_acc_{}.pth".format(best_acc * 10000))