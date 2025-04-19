import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune
import time, copy, os
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_train = datasets.CIFAR10(root="../data", train=True, download=True,
                               transform=transform)
cifar_test = datasets.CIFAR10(root="../data", train=False, download=True,
                              transform=transform)

train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=2)

print(f"CIFAR-10: {len(cifar_train)} train images, {len(cifar_test)} test images.")

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

num_epochs = 20

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

model, _ = train_model(device, num_epochs)

baseline_metrics = {}
def evaluate_model(model, data_loader, name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    total_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), f"{name}_model.pt")
    disk_size = os.path.getsize(f"{name}_model.pt") / 1024**2
    baseline_metrics[name] = {
        "accuracy": acc,
        "params": total_params,
        "disk_size_mb": disk_size
    }
    print(f"{name}: Accuracy = {acc:.2f}%, Parameters = {total_params/1e6:.2f}M, Disk Size = {disk_size:.2f} MB")

evaluate_model(model, test_loader, "CIFAR-10")

sparsity_levels = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pruning_results = {}

base_model = model.to(device)
base_model.eval()

base_params = sum(p.numel() for p in base_model.parameters())

for sparsity in sparsity_levels:
    print(f"\nPruning {int(sparsity*100)}% of weights...")
    model_pruned = copy.deepcopy(base_model)
    parameters_to_prune = [
        (m, "weight")
        for m in model_pruned.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity
    )

    for m, name in parameters_to_prune:
        prune.remove(m, name)

    total = sum(p.numel() for p in model_pruned.parameters())
    zeros = sum((p == 0).sum().item() for p in model_pruned.parameters())
    nonzeros = total - zeros
    global_s = zeros / total * 100

    correct = 0
    total_lbl = 0
    model_pruned.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, preds = torch.max(model_pruned(images), 1)
            correct += (preds == labels).sum().item()
            total_lbl += labels.size(0)
    acc = 100.0 * correct / total_lbl

    start = time.time()
    cnt = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model_pruned(images)
            cnt += images.size(0)
    thr = cnt / (time.time() - start)

    pruning_results[f"{sparsity}"] = {
        "accuracy": acc,
        "nonzero_params": nonzeros,
        "throughput": thr,
        "global_sparsity": global_s
    }
    print(f" → Acc: {acc:.2f}%, Params: {nonzeros} ({global_s:.1f}% sparsity), Thrpt: {thr:.2f} img/s")

levels_pct = [0, 10, 20, 30, 40, 50, 60, 70, 80]
acc_vals = [
    pruning_results.get(str(s))["accuracy"]
    for s in [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
]
thr_vals = [
    pruning_results.get(str(s))["throughput"]
    for s in [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
]

plt.figure(figsize=(8, 5))
plt.plot(levels_pct, acc_vals, marker="o")
plt.xlabel("Pruned Weights (%)")
plt.ylabel("Accuracy (%)")
plt.title("CIFAR-10: Accuracy vs Pruning Level")
plt.xticks(levels_pct)
plt.grid(True)
plt.savefig("cifar_accuracy_vs_pruning.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(levels_pct, thr_vals, marker="o")
plt.xlabel("Pruned Weights (%)")
plt.ylabel("Throughput (images/sec)")
plt.title("CIFAR-10: Throughput vs Pruning Level")
plt.xticks(levels_pct)
plt.grid(True)
plt.savefig("cifar_throughput_vs_pruning.png")
plt.close()