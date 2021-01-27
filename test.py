from timer import Timer
import time
import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from tasks.charts import savefig
def train(batchsize, epochs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    train_dataset = torchvision.datasets.ImageNet(
        root="/groups2/gaa50004/data/ILSVRC2012/pytorch",
        split="train",
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transform,
        ])
    )
    test_dataset = torchvision.datasets.ImageNet(
        root="/groups2/gaa50004/data/ILSVRC2012/pytorch",
        split="val",
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transform,
        ])
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize,
        shuffle=True, num_workers=4
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize,
        shuffle=False, num_workers=4
        )                                          


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timer = Timer()


    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print(len(train_loader))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    info = {'iter':[], 'loss':[]}
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            epoch_frac = epoch + i / len(train_loader)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with timer("forward", epoch_frac):
                outputs = model(inputs)
            with timer("backward", epoch_frac):
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # print statistics
            info['iter'].append(epoch_frac)
            info['loss'].append(loss.item())
            if (i==500):
                break
    
    savefig(info['iter'], info['loss'], name="test.png")
    print('Finished Training')
    print(timer.summary())
if __name__ == "__main__":
    train(batchsize=512, epochs=1)
