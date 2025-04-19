import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune
import time, copy, os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4) if False else transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
test_transform = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

cifar_train = datasets.CIFAR10(root="../data", train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914,0.4822,0.4465),
                                                        (0.2470,0.2435,0.2616))
                               ]))
cifar_test = datasets.CIFAR10(root="../data", train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914,0.4822,0.4465),
                                                       (0.2470,0.2435,0.2616))
                              ]))

cifar_train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=2)
cifar_test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=2)

imagenette_dir = "../imagenette"
imagenette_train = datasets.ImageFolder(os.path.join(imagenette_dir, "train"),
                                       transform=transforms.Compose([
                                           transforms.RandomResizedCrop(160),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485,0.456,0.406),
                                                                (0.229,0.224,0.225))
                                       ]))
imagenette_val = datasets.ImageFolder(os.path.join(imagenette_dir, "val"),
                                     transform=transforms.Compose([
                                         transforms.Resize(160),
                                         transforms.CenterCrop(160),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485,0.456,0.406),
                                                              (0.229,0.224,0.225))
                                     ]))
imagenette_train_loader = DataLoader(imagenette_train, batch_size=64, shuffle=True, num_workers=2)
imagenette_val_loader   = DataLoader(imagenette_val, batch_size=64, shuffle=False, num_workers=2)

imagewoof_dir = "../imagewoof"
imagewoof_train = datasets.ImageFolder(os.path.join(imagewoof_dir, "train"),
                                      transform=transforms.Compose([
                                          transforms.RandomResizedCrop(160),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485,0.456,0.406),
                                                               (0.229,0.224,0.225))
                                      ]))
imagewoof_val = datasets.ImageFolder(os.path.join(imagewoof_dir, "val"),
                                    transform=transforms.Compose([
                                        transforms.Resize(160),
                                        transforms.CenterCrop(160),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485,0.456,0.406),
                                                             (0.229,0.224,0.225))
                                    ]))
imagewoof_train_loader = DataLoader(imagewoof_train, batch_size=64, shuffle=True, num_workers=2)
imagewoof_val_loader   = DataLoader(imagewoof_val, batch_size=64, shuffle=False, num_workers=2)

print(f"CIFAR-10: {len(cifar_train)} train images, {len(cifar_test)} test images.")
print(f"ImageNette: {len(imagenette_train)} train images, {len(imagenette_val)} val images.")
print(f"ImageWoof: {len(imagewoof_train)} train images, {len(imagewoof_val)} val images.")

class VGG16(nn.Module):
    def __init__(self, num_classes=10, input_size=224):
        super(VGG16, self).__init__()
        cfg = [64, 64, "M", 
               128, 128, "M", 
               256, 256, 256, "M", 
               512, 512, 512, "M", 
               512, 512, 512, "M"]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            dummy_out = self.features(dummy)
            feat_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model_cifar    = VGG16(num_classes=10, input_size=32).to(device)
model_imagenette = VGG16(num_classes=10, input_size=160).to(device)
model_imagewoof = VGG16(num_classes=10, input_size=160).to(device)

def count_params(model):
    return sum(p.numel() for p in model.parameters())
print("VGG16 CIFAR-10 total params: %.2f million" % (count_params(model_cifar)/1e6))
print("VGG16 ImageNette total params: %.2f million" % (count_params(model_imagenette)/1e6))
print("VGG16 ImageWoof total params: %.2f million" % (count_params(model_imagewoof)/1e6))

epochs = 1
criterion = nn.CrossEntropyLoss()

def train_model(model, train_loader, val_loader, optimizer, epochs, model_name):
    best_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total

        if acc > best_acc:
            best_acc = acc
        print(f"{model_name} Epoch {epoch}: Loss = {epoch_loss:.4f}, Val Accuracy = {acc:.2f}%")
    print(f"Best {model_name} validation accuracy: {best_acc:.2f}%")
    return best_acc

optimizer_cifar = torch.optim.Adam(model_cifar.parameters(), lr=1e-3)
optimizer_imagenette = torch.optim.Adam(model_imagenette.parameters(), lr=1e-3)
optimizer_imagewoof = torch.optim.Adam(model_imagewoof.parameters(), lr=1e-3)

best_acc_cifar = train_model(model_cifar, cifar_train_loader, cifar_test_loader, optimizer_cifar, epochs, "CIFAR-10")
best_acc_imagenette = train_model(model_imagenette, imagenette_train_loader, imagenette_val_loader, optimizer_imagenette, epochs, "ImageNette")
best_acc_imagewoof = train_model(model_imagewoof, imagewoof_train_loader, imagewoof_val_loader, optimizer_imagewoof, epochs, "ImageWoof")

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

evaluate_model(model_cifar, cifar_test_loader, "CIFAR-10")
evaluate_model(model_imagenette, imagenette_val_loader, "ImageNette")
evaluate_model(model_imagewoof, imagewoof_val_loader, "ImageWoof")

sparsity_levels = [0.2, 0.5, 0.8]
pruning_results = { "CIFAR-10": {}, "ImageNette": {}, "ImageWoof": {} }

for dataset, base_model in [("CIFAR-10", model_cifar), 
                             ("ImageNette", model_imagenette), 
                             ("ImageWoof", model_imagewoof)]:
    print(f"\nPruning {dataset} model...")
    base_params = sum(p.numel() for p in base_model.parameters())
    base_nonzeros = base_params
    base_acc = baseline_metrics[dataset]["accuracy"]
    base_throughput = None
    pruning_results[dataset]["0.0 (baseline)"] = {
        "accuracy": base_acc,
        "nonzero_params": base_nonzeros,
        "throughput": None
    }

    model = base_model
    model.eval()
    start_time = time.time()
    total_images = 0
    with torch.no_grad():
        for images, labels in (cifar_test_loader if dataset=="CIFAR-10" else imagenette_val_loader if dataset=="ImageNette" else imagewoof_val_loader):
            images = images.to(device)
            outputs = model(images)
            total_images += images.size(0)
    elapsed = time.time() - start_time
    base_throughput = total_images / elapsed
    pruning_results[dataset]["0.0 (baseline)"]["throughput"] = base_throughput
    print(f"Baseline throughput: {base_throughput:.2f} images/sec")

    for sparsity in sparsity_levels:
        model_pruned = copy.deepcopy(base_model)
        model_pruned.eval()
        parameters_to_prune = []
        for name, module in model_pruned.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity
        )
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        total_params = 0
        total_zero = 0
        for param in model_pruned.parameters():
            total_params += param.numel()
            total_zero += torch.sum(param == 0).item()
        global_sparsity = total_zero / total_params
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in (cifar_test_loader if dataset=="CIFAR-10" else imagenette_val_loader if dataset=="ImageNette" else imagewoof_val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model_pruned(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total
        start_time = time.time()
        total_images = 0
        with torch.no_grad():
            for images, labels in (cifar_test_loader if dataset=="CIFAR-10" else imagenette_val_loader if dataset=="ImageNette" else imagewoof_val_loader):
                images = images.to(device)
                outputs = model_pruned(images)
                total_images += images.size(0)
        elapsed = time.time() - start_time
        throughput = total_images / elapsed
        key = str(sparsity)
        pruning_results[dataset][key] = {
            "accuracy": acc,
            "nonzero_params": total_params - total_zero,
            "throughput": throughput,
            "global_sparsity": global_sparsity
        }
        print(f"  Pruned {int(sparsity*100)}% -> Accuracy: {acc:.2f}%, Non-zero params: {total_params-total_zero} ({global_sparsity*100:.1f}% sparsity), Throughput: {throughput:.2f} img/s")

sparsity_ticks = [0, 0.2, 0.5, 0.8]

# Accuracy Plot
plt.figure(figsize=(8,5))
for dataset in ["CIFAR-10", "ImageNette", "ImageWoof"]:
    acc_vals = [pruning_results[dataset][str(s)]["accuracy"] if str(s) in pruning_results[dataset] 
                else pruning_results[dataset]["0.0 (baseline)"]["accuracy"] 
                for s in [0.0, 0.2, 0.5, 0.8]]
    plt.plot([0,20,50,80], acc_vals, marker="o", label=dataset)
plt.xlabel("Pruned Weights (%)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Pruning Level")
plt.xticks([0,20,50,80])
plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_pruning.png")
plt.close()

plt.figure(figsize=(8,5))
for dataset in ["CIFAR-10", "ImageNette", "ImageWoof"]:
    thr_vals = [pruning_results[dataset][str(s)]["throughput"] if str(s) in pruning_results[dataset] 
                else pruning_results[dataset]["0.0 (baseline)"]["throughput"] 
                for s in [0.0, 0.2, 0.5, 0.8]]
    plt.plot([0,20,50,80], thr_vals, marker="o", label=dataset)
plt.xlabel("Pruned Weights (%)")
plt.ylabel("Throughput (images/sec)")
plt.title("Throughput vs Pruning Level")
plt.xticks([0,20,50,80])
plt.legend()
plt.grid(True)
plt.savefig("throughput_vs_pruning.png")  # Save the plot
plt.close()  # Close the figure