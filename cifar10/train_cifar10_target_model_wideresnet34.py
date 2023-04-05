import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from cifar10 import CIFAR10_target_net
# from cifar10.wide_resnet import WideResNet as CIFAR10_target_net
from cifar10.WRN import WideResNet as CIFAR10_target_net


cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)


def testing_model(target_model):

    target_model.eval()

    # CIFAR10 test dataset
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    cifar10_dataset_test = torchvision.datasets.CIFAR10('./dataset', train=False, transform=test_transform,
                                                        download=True)
    test_dataloader = DataLoader(cifar10_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_lab == test_label, 0)
    accuracy = num_correct.item() / len(cifar10_dataset_test)
    print('accuracy in testing set: %f\n' % (accuracy))
    if accuracy > 0.92:
        # save model
        targeted_model_file_name = './CIFAR10_target_model_wrn.pth'
        torch.save(target_model.state_dict(), targeted_model_file_name)
        # import sys
        # sys.exit(0)


if __name__ == "__main__":
    use_cuda = True
    image_nc = 3
    batch_size = 256

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    cifar10_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=train_transform, download=True)
    train_dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # target_model = CIFAR10_target_net(depth=28, num_classes=10, widen_factor=20).to(device)
    # checkpoint = torch.load('./WRN_9575_cifar10_ckpt.pth')
    # target_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    # target_model.eval()
    # testing_model(target_model)


    # training the target model
    target_model = CIFAR10_target_net().to(device)
    target_model.train()
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.01)
    epochs = 2000
    for epoch in range(epochs):
        loss_epoch = 0
        if epoch == 20:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.01)
        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            loss_model = F.cross_entropy(logits_model, train_labels)
            loss_epoch += loss_model
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()

        print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))
        if epoch > 5:
            testing_model(target_model)
