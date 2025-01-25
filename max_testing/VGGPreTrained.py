from torchvision.models import vgg11, vgg13, vgg16, vgg19
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch
from torch.utils.data import Subset
import numpy as np
import time, datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

train_data_percentage = 0.3
architecture = 11
num_epochs = 20

pretrained_vgg = None
if architecture == 11:
    pretrained_vgg = vgg11(pretrained=True)
elif architecture == 13:
    pretrained_vgg = vgg13(pretrained=True)
elif architecture == 16:
    pretrained_vgg = vgg16(pretrained=True)
elif architecture == 19:
    pretrained_vgg = vgg19(pretrained=True)

f = open("../logs/CIFAR10RunTime.txt", "a")

print("writing")
f.write(f"===============================================================================\n")
f.write(f"Max Testing: Ran at {datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}\n")
f.write(f"Running on Pretrained VGG {architecture} Architecture, {int(train_data_percentage * 100)}% CIFAR-10 Training Data, and {num_epochs} Epoch(s)\n")
print("done writing")

pretrained_vgg.classifier[6] = nn.Linear(pretrained_vgg.classifier[6].in_features, 10)

for param in pretrained_vgg.features.parameters():
    param.requires_grad = False

device = torch.device("cuda")
pretrained_vgg = pretrained_vgg.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_vgg.classifier.parameters(), lr=0.001)

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

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    print("training start")
    model.train()
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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

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

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", class_report)

    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(class_report + "\n")

def format_time(time):
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)

    return f"{hours}h {minutes}m {seconds}s"

start_time = time.time()
train_model(pretrained_vgg, train_loader, criterion, optimizer, device, num_epochs)
end_time = time.time()

formatted_time = format_time(end_time - start_time)
print(f"GPU Training Time: {formatted_time}")
f.write(f"GPU Training Time: {formatted_time}\n")

test_model(pretrained_vgg, device)
f.write(f"===============================================================================")
f.write("\n")