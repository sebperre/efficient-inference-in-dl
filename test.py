import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import math
import sys

sys.path.append("/home/sebperre/programming-projects/efficient-inference-in-dl/utils")
from file_utils import write_file, print_write, get_args, timer

# Followed Geeksforgeeks description: https://www.geeksforgeeks.org/vgg-net-architecture-explained/

# Followed the format of https://github.com/chengyangfu/pytorch-vgg-cifar10
def make_layers(layer_config):
    layers = []
    in_channels = 3
    for v in layer_config:
        if v == "Pool":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, depth=1, num_classes = 10):
        super(VGG, self).__init__()
        self.features = make_layers(layer_configs[depth])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(depth_connections[depth], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        # Initial Weights Matter a lot, took this from https://github.com/chengyangfu/pytorch-vgg-cifar10
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # VGG Paper talks about this except with mean 0 and variance 10^-2
                # VGG Paper of mean 0 and variance 10^-2 gives vanishing gradients

                # This is Kaiming initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

                # This variance works though
                # m.weight.data.normal_(0, 0.025)
                # m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
@timer
def train_model(device, num_epochs, iteration):
    model = VGG(iteration).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.1, weight_decay=0.0005)

    running_loss = 0.0
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

        print(f"Iteration {iteration}: Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    print_write(f"Loss on Last Iteration for Training is {running_loss}")

    return model

@timer
def test_model(model, device):
    global label_tester
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
    
    if label_tester is None:
        label_tester = np.array(all_labels)

    predictions.append(np.array(all_preds) == label_tester)

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

def compare_overlap():
    global label_tester
    num_models = len(predictions)

    correct_counts = {}

    for i in range(num_models):
        for j in range(i + 1, num_models):
            percentage = None
            if np.sum(predictions[j]) > 0:
                percentage = (np.sum(predictions[j] & (predictions[i] == predictions[j])) / np.sum(predictions[j])) * 100
            else:
                percentage = 0

            correct_counts[(i, j)] = percentage
    print_write("Overlap:")
    print_write(f"{correct_counts}")

def setup():
    train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def execute():
    if torch.cuda.is_available():
        print("Training on GPU...")
        gpu_device = torch.device("cuda")
        for iteration in range(1, iterations + 1):
            model = train_model(gpu_device, num_epochs, iteration, description=f"Training Iteration {iteration}")
            print_write(f"\nIteration {iteration}: Testing on Test Set")
            test_model(model, gpu_device, description=f"Testing Iteration {iteration}")
        
        compare_overlap()
    else:
        print("GPU not available.")

if __name__ == "__main__":
    layer_configs = {
        1: [64, "Pool"],
        2: [64, 64, "Pool"],
        3: [64, 64, "Pool", 128, 128, "Pool"],
        4: [64, 64, 64, "Pool", 128, 128, 128, "Pool"],
        5: [64, 64, 64, "Pool", 128, 128, 128, "Pool", 256, 256, 256, "Pool"]
    }

    depth_connections = {
        1: 16384,
        2: 16384,
        3: 8192,
        4: 8192,
        5: 4096
    }

    iterations = len(depth_connections)

    predictions = []
    label_tester = None
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    for inputs, labels in test_loader:
        print(labels == np.array([1 for x in range(64)]))
        break

    # for inputs, labels in train_loader:
    #     print(inputs)

    # args = get_args(epoch=True)
    # num_epochs = args.epochs
    # train_loader, test_loader = setup()
    # f = write_file("iterative_models")
    # f.write("Using VGG Model\n")
    # execute()
