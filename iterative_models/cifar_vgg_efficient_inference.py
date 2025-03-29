import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import math
import sys
import time

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
    
# class ImageToModelClassifier(nn.module):
    

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
def train_oracle_model(model, device):
    global correct_labels
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
    
    if correct_labels is None:
        correct_labels = np.array(all_labels)

    predictions.append(np.array(all_preds))

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    class_report = classification_report(all_labels, all_preds)

    print_write(f"Accuracy: {accuracy:.4f}")
    print_write(f"Precision: {precision:.4f}")
    print_write(f"Recall: {recall:.4f}")
    print_write(f"F1 Score: {f1:.4f}")
    print_write("\nClassification Report:\n")
    print_write(class_report)

def get_oracle_percentages():
    global correct_labels
    num_models = len(predictions)
    
    print_write("Overlap Table\n")

    col_width1 = 12
    col_width2 = 25
    col_width3 = 25

    oracle_percentages = {}

    for w in range(num_models):
        for s in range(w + 1, num_models):
            print_write(f"Weaker {w} and Stronger {s}")
            print_write(f"{"Label":<{col_width1}}{"% Weaker of Stronger":<{col_width2}}{"% Stronger of Weaker":<{col_width3}}")
            print_write("-" * (col_width1 + col_width2 + col_width3))
            
            percent_comparison = []

            for i in range(10):
                total_correct_strong = np.count_nonzero((correct_labels == i) & (predictions[s] == correct_labels))
                total_correct_weak = np.count_nonzero((correct_labels == i) & (predictions[w] == correct_labels))
                total_correct_both = np.count_nonzero((correct_labels == i) & (predictions[w] == correct_labels) & (predictions[s] == correct_labels))

                if total_correct_strong == 0:
                    percent_weaker_of_stronger = 0
                else:
                    percent_weaker_of_stronger = round((total_correct_both / total_correct_strong) * 100, 2)
                    percent_comparison.append(percent_weaker_of_stronger)
                if total_correct_weak == 0:
                    percent_stronger_of_weaker = 0
                else:
                    percent_stronger_of_weaker = round((total_correct_both / total_correct_weak) * 100, 2)
                print_write(f"{i:<{col_width1}}{percent_weaker_of_stronger:<{col_width2}}{percent_stronger_of_weaker:<{col_width3}}")
            print_write("-" * (col_width1 + col_width2 + col_width3))
            
            oracle_percentages[(w, s)] = percent_comparison

            total_correct_strong = np.count_nonzero(predictions[s] == correct_labels)
            total_correct_weak = np.count_nonzero(predictions[w] == correct_labels)
            total_correct_both = np.count_nonzero((predictions[w] == correct_labels) & (predictions[s] == correct_labels))

            if total_correct_strong == 0:
                percent_weaker_of_stronger = 0
            else:
                percent_weaker_of_stronger = round((total_correct_both / total_correct_strong) * 100, 2)
            if total_correct_weak == 0:
                percent_stronger_of_weaker = 0
            else:
                percent_stronger_of_weaker = round((total_correct_both / total_correct_weak) * 100, 2)
            print_write(f"% Weaker of Stronger Total: {percent_weaker_of_stronger}")
            print_write(f"% Stronger of Weaker Total: {percent_stronger_of_weaker}")
            print_write("")

    return oracle_percentages

@timer
def test_combined_model(models, model_classifier, device):
    all_preds = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            for input, label in zip(inputs, labels):
                model_classifier_output = model_classifier(torch.unsqueeze(input, 0))
                _, model_pred = torch.max(model_classifier_output, 1)
                output = models[model_pred.item()](torch.unsqueeze(input, 0))
                _, pred = torch.max(output, 1)

                all_preds.append(int(pred.cpu().numpy()))
                all_labels.append(int(label.cpu().numpy()))
    end_time = time.time()

    return all_labels, all_preds, end_time - start_time

@timer
def test_best_model(model, device):
    model.eval()
    all_preds = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            for input, label in zip(inputs, labels):
                output = model(torch.unsqueeze(input, 0))
                _, pred = torch.max(output, 1)

                all_preds.append(int(pred.cpu().numpy()))
                all_labels.append(int(label.cpu().numpy()))
    end_time = time.time()

    return all_labels, all_preds, end_time - start_time

def compare_models(combined_all_labels, combined_all_preds, combined_time, best_all_labels, best_all_preds, best_model_time):
    print_write("Model Comparison\n")

    col_width1 = 20
    col_width2 = 25
    col_width3 = 25
    col_width4 = 25

    accuracy_best = round(accuracy_score(best_all_labels, best_all_preds) * 100, 4)
    precision_best = round(precision_score(best_all_labels, best_all_preds, average="weighted") * 100, 4)
    recall_best = round(recall_score(best_all_labels, best_all_preds, average="weighted") * 100, 4)
    f1_best = round(f1_score(best_all_labels, best_all_preds, average="weighted") * 100, 4)
    class_report_best = classification_report(best_all_labels, best_all_preds)

    accuracy_combined = round(accuracy_score(combined_all_labels, combined_all_preds) * 100, 4)
    precision_combined = round(precision_score(combined_all_labels, combined_all_preds, average="weighted") * 100, 4)
    recall_combined = round(recall_score(combined_all_labels, combined_all_preds, average="weighted") * 100, 4)
    f1_combined = round(f1_score(combined_all_labels, combined_all_preds, average="weighted") * 100, 4)
    class_report_combined = classification_report(combined_all_labels, combined_all_preds)

    print_write(f"{"Statistic":<{col_width1}}{"Best":<{col_width2}}{"Combined":<{col_width3}}{"Difference":<{col_width4}}")
    print_write("-" * (col_width1 + col_width2 + col_width3 + col_width4))
    print_write(f"{"Accuracy (%)":<{col_width1}}{accuracy_best:<{col_width2}}{accuracy_combined:<{col_width3}}{round(accuracy_best - accuracy_combined, 4):<{col_width4}}")
    print_write(f"{"Precision (%)":<{col_width1}}{precision_best:<{col_width2}}{precision_combined:<{col_width3}}{round(precision_best - precision_combined, 4):<{col_width4}}")
    print_write(f"{"Recall (%)":<{col_width1}}{recall_best:<{col_width2}}{recall_combined:<{col_width3}}{round(recall_best - recall_combined, 4):<{col_width4}}")
    print_write(f"{"F1 (%)":<{col_width1}}{f1_best:<{col_width2}}{f1_combined:<{col_width3}}{round(f1_best - f1_combined, 4):<{col_width4}}")
    print_write(f"{"Time (s)":<{col_width1}}{round(best_model_time, 4):<{col_width2}}{round(combined_time, 4):<{col_width3}}{round(best_model_time - combined_time, 4):<{col_width4}}")
    print_write("-" * (col_width1 + col_width2 + col_width3 + col_width4))

    print_write("")
    print_write("Best Model Class Report")
    print_write(class_report_best)

    print_write("")
    print_write("Combined Model Class Report")
    print_write(class_report_combined)

def setup():
    dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
    train_dataset, oracle_train_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    oracle_train_loader = torch.utils.data.DataLoader(oracle_train_dataset, batch_size=64, shuffle=False)

    test_dataset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, oracle_train_loader, test_loader

@timer
def train_model_classifier(device, num_epochs, model_association):
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(in_features=1024, out_features=5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)

            labels = torch.tensor([model_association[label.item()] for label in labels], dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    return model

def execute():
    if torch.cuda.is_available():
        print("Training on GPU...")
        models = {}
        gpu_device = torch.device("cuda")
        for iteration in range(1, iterations + 1):
            model = train_model(gpu_device, num_epochs, iteration, description=f"Training Iteration {iteration}")
            models[iteration - 1] = model
            print_write(f"\nIteration {iteration}: Testing on Test Set")
            train_oracle_model(model, gpu_device, description=f"Testing Iteration {iteration}")
        
        oracle_percentages = get_oracle_percentages()
        oracles_for_testing = {(x, 4): oracle_percentages[(x, 4)] for x in range(4)}

        model_association = {}
        for i in range(10):
            model_association[i] = 4
            for j in range(4):
                if oracles_for_testing[(j, 4)][i] >= (1 - acc_sac) * 100:
                    model_association[i] = j
                    break

        model_classifier_epochs = 15
        model_classifier = train_model_classifier(gpu_device, model_classifier_epochs, model_association, description=f"Training Model Classifier")

        for model in models.values():
            model.eval()
        model_classifier.eval()

        combined_all_labels, combined_all_preds, combined_time = test_combined_model(models, model_classifier, gpu_device, description=f"Testing Oracle")
        best_all_labels, best_all_preds, best_model_time = test_best_model(models[4], gpu_device, description=f"Testing Best Model")

        compare_models(combined_all_labels, combined_all_preds, combined_time, best_all_labels, best_all_preds, best_model_time)
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

    iterations = len(layer_configs)

    predictions = []
    correct_labels = None
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    args = get_args(epoch=True, acc_sac=True)
    num_epochs = args.epochs
    acc_sac = args.acc_sac
    train_loader, oracle_train_loader, test_loader = setup()
    f, _ = write_file("iterative_models", "CIFAR-10", "Mini-VGGs", num_epochs, None, acc_sac)
    f.write("Using VGG Model\n")
    execute()
