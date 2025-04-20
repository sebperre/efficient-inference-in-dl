import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import sys
import os

sys.path.append("/home/sebperre/programming-projects/efficient-inference-in-dl/utils")

from file_utils import write_file, print_write, get_args, timer
from subset_data import get_subset
from visualizations import plot_loss_per_epoch

PATH = None

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

@timer
def train_model(device, num_epochs, description="Training"):
    model = VGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.1, weight_decay=0.0005)

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
def test_model(model, device, description="Testing"):
    model.eval()
    all_preds, all_labels = [], []

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
    data_dir = "../imagewoof"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)

    if subset_size != -1:
        train_dataset = get_subset(train_dataset, subset_size=subset_size)
        test_dataset = get_subset(test_dataset, subset_size=subset_size)

    global train_loader, test_loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, test_loader

def execute():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print("Training on GPU...")
        gpu_device = torch.device("cuda")
        model, train_losses = train_model(gpu_device, num_epochs, description="Training")
        test_model(model, gpu_device, description="Testing")
        plot_loss_per_epoch(num_epochs, train_losses, "Training Loss Per Epoch (ImageWoof VGG-16)", f"{PATH}/loss_per_epoch")
    else:
        print("GPU not available.")

if __name__ == "__main__":
    args = get_args(epoch=True, subset=True)
    num_epochs = args.epochs
    subset_size = args.subset
    train_loader, test_loader = setup()
    f, PATH = write_file("max_testing", "ImageWoof", "VGG-16", num_epochs, subset_size)
    f.write("Using VGG Model on ImageWoof\n")
    execute()