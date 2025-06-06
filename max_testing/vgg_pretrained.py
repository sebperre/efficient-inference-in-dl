from torchvision.models import vgg11, vgg13, vgg16, vgg19
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch
from torch.utils.data import Subset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import sys

sys.path.append("/home/sebperre/programming-projects/efficient-inference-in-dl/utils")

from file_utils import write_file, print_write, get_args, timer
from visualizations import plot_loss_per_epoch

PATH = None

@timer
def train_model(model, train_loader, criterion, optimizer, device, num_epochs, description="Training"):
    print("training start")
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
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

    return epoch_losses

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
    print_write("\nClassification Report:\n")
    print_write(class_report)

def setup():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)

    subset_size = int(len(train_dataset) * train_data_percentage)
    indices = np.random.choice(len(train_dataset), size=subset_size, replace=False)

    train_subset = Subset(train_dataset, indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

    test_dataset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def execute():
    pretrained_vgg = None
    if architecture == 11:
        pretrained_vgg = vgg11(pretrained=True)
    elif architecture == 13:
        pretrained_vgg = vgg13(pretrained=True)
    elif architecture == 16:
        pretrained_vgg = vgg16(pretrained=True)
    elif architecture == 19:
        pretrained_vgg = vgg19(pretrained=True)

    pretrained_vgg.classifier[6] = nn.Linear(pretrained_vgg.classifier[6].in_features, 10)

    for param in pretrained_vgg.features.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_vgg = pretrained_vgg.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_vgg.classifier.parameters(), lr=0.001)

    train_losses = train_model(pretrained_vgg, train_loader, criterion, optimizer, device, num_epochs, description="Training")
    test_model(pretrained_vgg, device, description="Testing")
    plot_loss_per_epoch(num_epochs, train_losses, f"Training Loss Per Epoch (Pretrained VGG-{architecture})", f"{PATH}/loss_per_epoch")

if __name__ == "__main__":
    train_data_percentage = 0.3
    architecture = 11
    args = get_args(epoch=True)
    num_epochs = args.epochs
    train_loader, test_loader = setup()
    f, PATH = write_file("max_testing", "CIFAR-10", f"Pretrained VGG-{architecture}", num_epochs)
    f.write(f"Using Pretrained VGG-{architecture} on CIFAR-10\n")
    execute()