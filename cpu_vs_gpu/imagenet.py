import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import time
import sys

sys.path.append("/home/sebperre/programming-projects/efficient-inference-in-dl/utils")

from file_utils import write_file, print_write, get_args, timer
from subset_data import get_subset

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1000)

    def forward(self, x):
        return self.model(x)

def train_model(device, num_epochs, train_loader):
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
    cpu_time = train_model(cpu_device, num_epochs, train_loader)
    print_write(f"GPU Training Time: {cpu_time:.2f} seconds")

def run_on_gpu():
    if torch.cuda.is_available():
        print("Training on GPU...")
        gpu_device = torch.device("cuda")
        gpu_time = train_model(gpu_device, num_epochs, train_loader)
        print_write(f"GPU Training Time: {gpu_time:.2f} seconds")
    else:
        print("GPU not available.")

def setup():
    path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")

    print("CPU vs. GPU Testing: Path to dataset files:", path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_dir = "~/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini"
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)

    train_subset = get_subset(train_dataset, subset_size=subset_size)
    return torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

@timer  
def execute():
    run_on_cpu()
    run_on_gpu()

if __name__ == "__main__":
    args = get_args(epoch=True, subset=True)
    num_epochs = args.epochs
    subset_size = args.subset
    train_loader = setup()
    f, _ = write_file("cpu_vs_gpu")
    f.write("Using ResNet Model\n")
    execute(description="CPU and GPU on ImageNet with a ResNet model")
