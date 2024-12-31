import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import kagglehub
import os 

path = kagglehub.dataset_download("sanikamal/horses-or-humans-dataset")

# Set random seed for reproducibility
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = HorseOrHuman(data_dir, transform)

class HorseOrHumanDetect(nn.Module):
    def __init__(self):
        super(HorseOrHumanDetect, self).__init__()
        self.model = torchvision.models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x):
        return self.model(x)

model = HorseOrHumanDetect()

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)

train_folder = os.path.join(path, 'horse-or-human', 'train')
test_folder = os.path.join(path, 'horse-or-human', 'validation')

train_data = HorseOrHuman(train_folder, transform=transform)
test_data = HorseOrHuman(test_folder, transform=transform)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32)

epochs = 20

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)  # Move data to GPU
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Training Loss: {total_loss / len(train_dataloader)}")

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)  # Move data to GPU
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            _, predicted = torch.max(pred, dim=1)
            correct += (predicted == y).sum().item()
    accuracy = correct / len(test_data)
    print(f"Test Loss: {test_loss / len(test_dataloader)}")
    print(f"Test Accuracy: {accuracy:.4f}")
