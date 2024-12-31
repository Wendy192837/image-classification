import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import kagglehub
import os 

path = kagglehub.dataset_download("sanikamal/horses-or-humans-dataset")
# print("Path to dataset files:", path)

torch.manual_seed(42)

class HorseOrHuman(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = torchvision.datasets.ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id]

    def classes(self):
        return self.data.classes

data_dir = path
dataset = HorseOrHuman(data_dir=data_dir)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = HorseOrHuman(data_dir, transform)

batch_size = 32

class HorseOrHumanDetect(nn.Module):
    def __init__(self):
        super(HorseOrHumanDetect, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1) 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 64 * 64, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
model = HorseOrHumanDetect()
# print(str(model)[:500])

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

model = HorseOrHumanDetect()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)

train_folder = os.path.join(path, 'horse-or-human', 'train')
test_folder = os.path.join(path, 'horse-or-human', 'validation')

train_data = HorseOrHuman(train_folder, transform=transform)
test_data = HorseOrHuman(test_folder, transform=transform)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32)

epochs = 10

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss, optimizer)
    test_loop(test_dataloader, model, loss)

classes = ["horse", "human"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()

wrong = 0
correct = 0

for i in range(100):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        x = x.unsqueeze(0)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        probabilities = torch.nn.functional.softmax(pred[0], dim=0)
        if (predicted != actual):
            wrong += 1
            img = test_data[i][0].squeeze(0) 
            img = transforms.ToPILImage()(img)
            # img.show()
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
            prob, catid = torch.topk(probabilities, 2)
            for i in range(prob.size(0)):
                print(classes[catid[i]], prob[i].item())
        else: correct += 1
print("wrong: ", wrong)
print("correct: ", correct)