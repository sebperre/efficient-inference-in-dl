import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import datetime
import os

f = open("../logs/CIFAR10RunTime/CIFAR10RunTime.txt", "a")

f.write(f"CPU vs. GPU Testing: Ran at {datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}\n")
num_epochs = 1

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

def train_model(device, num_epochs):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    end_time = time.time()
    return end_time - start_time

# Run on CPU
print("Training on CPU...")
cpu_device = torch.device("cpu")
f.write(f"Running on {num_epochs} epoch(s)\n")
cpu_time = train_model(cpu_device, num_epochs)
print(f"CPU Training Time: {cpu_time:.2f} seconds")
f.write(f"CPU Training Time: {cpu_time:.2f} seconds\n")

# Run on GPU
if torch.cuda.is_available():
    print("Training on GPU...")
    gpu_device = torch.device("cuda")
    gpu_time = train_model(gpu_device, num_epochs)
    print(f"GPU Training Time: {gpu_time:.2f} seconds")
    f.write(f"GPU Training Time: {gpu_time:.2f} seconds\n")
else:
    print("GPU not available.")

f.write(f"\n")
