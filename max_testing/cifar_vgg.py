# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# import math
# import sys
# sys.path.append("/home/sebperre/programming-projects/efficient-inference-in-dl/utils")

# from file_utils import write_file, get_args, timer, print_write

# # Followed Geeksforgeeks description: https://www.geeksforgeeks.org/vgg-net-architecture-explained/

# class VGG(nn.Module):
#     def __init__(self, num_classes = 10):
#         super(VGG, self).__init__()
#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 2
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 3
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 4
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 5
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(512, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )

#         # Initial Weights matter a lot, took this from https://github.com/chengyangfu/pytorch-vgg-cifar10
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # VGG Paper talks about this except with mean 0 and variance 10^-2
#                 # VGG Paper of mean 0 and variance 10^-2 gives vanishing gradients

#                 # This is Kaiming initialization
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 m.bias.data.zero_()

#                 # This variance works though
#                 # m.weight.data.normal_(0, 0.025)
#                 # m.bias.data.zero_()

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# @timer
# def train_model(device, num_epochs):
#     model = VGG().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.1, weight_decay=0.0005)

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

#     return model

# @timer
# def test_model(model, device):
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='weighted')
#     recall = recall_score(all_labels, all_preds, average='weighted')
#     f1 = f1_score(all_labels, all_preds, average='weighted')
#     class_report = classification_report(all_labels, all_preds)

#     print_write(f"Accuracy: {accuracy:.4f}")
#     print_write(f"Precision: {precision:.4f}")
#     print_write(f"Recall: {recall:.4f}")
#     print_write(f"F1 Score: {f1:.4f}")
#     print_write("\nClassification Report:")
#     print_write(class_report)


# def setup():
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

#     test_dataset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
#     return train_loader, test_loader

# def execute():
#     if torch.cuda.is_available():
#         print("Training on GPU...")
#         gpu_device = torch.device("cuda")
#         model = train_model(gpu_device, num_epochs, description="Training")
#         print("\nTesting on Test Set")
#         test_model(model, gpu_device, description="Testing")
#     else:
#         print("GPU not available.")

# if __name__ == "__main__":
#     args = get_args(epoch=True)
#     num_epochs = args.epochs
#     train_loader, test_loader = setup()
#     f, _ = write_file("max_testing", "CIFAR-10", "VGG-16", num_epochs)
#     execute()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import math
import sys
sys.path.append("/home/sebperre/programming-projects/efficient-inference-in-dl/utils")

from file_utils import write_file, get_args, timer, print_write
from visualizations import plot_loss_per_epoch

PATH = None

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def execute():
    if torch.cuda.is_available():
        print("Training on GPU...")
        gpu_device = torch.device("cuda")
        model, train_losses = train_model(gpu_device, num_epochs, description="Training")
        print("\nTesting on Test Set")
        test_model(model, gpu_device, description="Testing")
        plot_loss_per_epoch(num_epochs, train_losses, "Training Loss Per Epoch (CIFAR-10 VGG-16)", f"{PATH}/loss_per_epoch")
    else:
        print("GPU not available.")

if __name__ == "__main__":
    args = get_args(epoch=True)
    num_epochs = args.epochs
    train_loader, test_loader = setup()
    f, PATH = write_file("max_testing", "CIFAR-10", "VGG-16", num_epochs)
    f.write("Using VGG-16 Model\n")
    execute()
