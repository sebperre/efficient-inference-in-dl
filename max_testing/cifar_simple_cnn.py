import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/sebperre/programming-projects/efficient-inference-in-dl/utils")
from file_utils import write_file, get_args, timer, print_write
from visualizations import plot_loss_per_epoch

PATH = None

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

@timer
def train_model(device, num_epochs):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []

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

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model, epoch_losses

@timer
def test_model(model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    class_report = classification_report(all_labels, all_preds)

    print_write(f"Accuracy: {accuracy:.4f}")
    print_write(f"Precision: {precision:.4f}")
    print_write(f"Recall: {recall:.4f}")
    print_write(f"F1 Score: {f1:.4f}")
    print_write("\nClassification Report:")
    print_write(class_report)

def setup():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def execute():
    if torch.cuda.is_available():
        print("Training on GPU...")
        gpu_device = torch.device("cuda")
        model, train_losses = train_model(gpu_device, num_epochs, description="Training")
        print("\nTesting on Test Set")
        test_model(model, gpu_device, description="Testing")
        plot_loss_per_epoch(num_epochs, train_losses, "Training Loss Per Epoch (CIFAR-10 Simple CNN)", f"{PATH}/loss_per_epoch")
    else:
        print("GPU not available.")

if __name__ == "__main__":
    args = get_args(epoch=True)
    num_epochs = args.epochs
    train_loader, test_loader = setup()
    f, PATH = write_file("max_testing", "CIFAR-10", "Simple CNN", num_epochs)
    f.write("Using Simple CNN Model\n")
    execute()