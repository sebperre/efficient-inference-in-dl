import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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

@timer
def train_model(device, num_epochs, train_loader):
    model = SimpleResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    return model

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

    print_write(f"Accuracy: {accuracy:.4f}\n")
    print_write(f"Precision: {precision:.4f}\n")
    print_write(f"Recall: {recall:.4f}\n")
    print_write(f"F1 Score: {f1:.4f}\n")
    print_write("\nClassification Report:")
    print_write(class_report)

def setup():
    path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_dir = "~/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini"
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

    if subset_size != -1:
        train_dataset = get_subset(train_dataset, subset_size=subset_size)
        test_dataset = get_subset(test_dataset, subset_size=subset_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader
 
def execute():
    if torch.cuda.is_available():
        print("Training on GPU...")
        gpu_device = torch.device("cuda")
        model = train_model(gpu_device, num_epochs, train_loader, description="Training")
        test_model(model, gpu_device, description="Testing")
    else:
        print("GPU not available.")

if __name__ == "__main__":
    args = get_args(epoch=True, subset=True)
    num_epochs = args.epochs
    subset_size = args.subset
    train_loader, test_loader = setup()
    f = write_file("max_testing")
    f.write("Using ResNet Model\n")
    execute()
