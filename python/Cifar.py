import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import sys


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    model = Model().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device),data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # final accuracy: 56 %

def predict(model,img_path):
    img=Image.open(img_path).resize((32,32))
    img_tns=torchvision.transforms.ToTensor()(img).unsqueeze(0)
    # img_tns=transform(img).unsqueeze(0)

    result=model(img_tns) 
    probabilities=torch.softmax(result[0],dim=0)
    # class_index=torch.argmax(result)
    max_val,max_indx=torch.max(probabilities,dim=0)
    item=classes[max_indx]
    print(f'{img_path} --> {item} : {max_val*100:0.2f}% : index {max_indx}')

def save(model):
    model.eval()
    input=torch.randn(1,3,32,32)
    m=torch.jit.trace(model.to(device),input)
    torch.jit.save(m,'m_cifar.pt')
    # s=optimize_for_mobile(m)
    # s._save_for_lite_interpreter('m10.ptl') # this prediction is not correct

# load saved model
def load_model(path):
    model=torch.jit.load(path)
    model.eval()
    # count=sum(p.numel() for p in model.parameters())
    # print(f"prams: {count}")
    return model
def test_model():
    model=load_model("m_cifar.pt")
    imgs=['plane.jpg','car.jpeg','cat.jpg','deer.jpeg','dog.jpg','hen.jpeg']
    for img in imgs:
        predict(model,img)

# model=load_model('m_cifar.pt')
# save(model)
# train()
test_model()





