import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import numpy as np
import time
import datetime
import os

from utils import create_log_dir_and_folder, write_log_file, print_write

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1000)

    def forward(self, x):
        return self.model(x)

def get_subset(dataset, subset_size=1000, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    return Subset(dataset, indices)

def train_model(device, num_epochs):
    model = SimpleResNet().to(device)
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

def run_on_cpu():
    print("Training on CPU...")
    cpu_device = torch.device("cpu")
    f.write(f"Running on {num_epochs} epoch(s) and {subset_size} images\n")
    cpu_time = train_model(cpu_device, num_epochs)
    print(f"CPU Training Time: {cpu_time:.2f} seconds")
    f.write(f"CPU Training Time: {cpu_time:.2f} seconds\n")

def run_on_gpu():
    if torch.cuda.is_available():
        print("Training on GPU...")
        gpu_device = torch.device("cuda")
        gpu_time = train_model(gpu_device, num_epochs)
        print(f"GPU Training Time: {gpu_time:.2f} seconds")
        f.write(f"GPU Training Time: {gpu_time:.2f} seconds\n")
    else:
        print("GPU not available.")

def setup():
    path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")

    print("CPU vs. GPU Testing: Path to dataset files:", path)

    f.write(f"Ran at {datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}\n")
    num_epochs = 1
    subset_size = 1000

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_dir = "~/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini"
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)

    train_subset = get_subset(train_dataset, subset_size=subset_size)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

if __name__ == "__main__":
    create_log_dir_and_folder()
    f = open("../logs/ImageNetTime.txt", "a")
    setup()
    run_on_cpu()
    run_on_gpu()

    f.write(f"\n")