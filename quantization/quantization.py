import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant
import os
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.qconfig import QConfig
import time

print(torch.backends.quantized.supported_engines)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_set_cifar = datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transforms)
test_set_cifar  = datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transforms)
train_loader_cifar = torch.utils.data.DataLoader(train_set_cifar, batch_size=8, shuffle=True)
test_loader_cifar  = torch.utils.data.DataLoader(test_set_cifar, batch_size=8, shuffle=False)

train_dir_inette = "../imagenette/train"
val_dir_inette = "../imagenette/val"
train_set_inette = datasets.ImageFolder(train_dir_inette, transform=train_transforms)
test_set_inette  = datasets.ImageFolder(val_dir_inette, transform=test_transforms)
train_loader_inette = torch.utils.data.DataLoader(train_set_inette, batch_size=8, shuffle=True)
test_loader_inette  = torch.utils.data.DataLoader(test_set_inette, batch_size=8, shuffle=False)

train_dir_iwoof = "../imagewoof/train"
val_dir_iwoof = "../imagewoof/val"
train_set_iwoof = datasets.ImageFolder(train_dir_iwoof, transform=train_transforms)
test_set_iwoof  = datasets.ImageFolder(val_dir_iwoof, transform=test_transforms)
train_loader_iwoof = torch.utils.data.DataLoader(train_set_iwoof, batch_size=8, shuffle=True)
test_loader_iwoof  = torch.utils.data.DataLoader(test_set_iwoof, batch_size=8, shuffle=False)

def train_model(model, train_loader, val_loader, device, epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(1, epochs+1):
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
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        val_acc = 100.0 * correct / len(val_loader.dataset)
        print(f"Epoch {epoch}: Loss = {epoch_loss:.3f}, Val Acc = {val_acc:.2f}%")
        model.train()
    #return val_acc

def initialize_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    model.classifier[6] = nn.Linear(4096, 10)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier[:6].parameters():
        param.requires_grad = False

    return model

model_cifar = initialize_model()
model_inette = initialize_model()
model_iwoof = initialize_model()

acc_cifar  = train_model(model_cifar, train_loader_cifar, test_loader_cifar, device, epochs=0)
acc_inette = train_model(model_inette, train_loader_inette, test_loader_inette, device, epochs=0)
acc_iwoof  = train_model(model_iwoof, train_loader_iwoof, test_loader_iwoof, device, epochs=0)
#print("Final accuracy: CIFAR-10 = %.2f%%, ImageNette = %.2f%%, ImageWoof = %.2f%%" 
#      % (acc_cifar, acc_inette, acc_iwoof))

def quantize_model(model, data_loader_calib, device):
    model.eval()
    for module_name, module in model.named_children():
        if module_name == "features":
            fuse_list = []
            last_conv = None
            for name, layer in module._modules.items():
                if isinstance(layer, nn.Conv2d):
                    last_conv = name
                if isinstance(layer, nn.ReLU) and last_conv:
                    fuse_list.append([last_conv, name])
                    last_conv = None
            quant.fuse_modules(module, fuse_list, inplace=True)
        elif module_name == "classifier":
            fuse_list = []
            last_linear = None
            for name, layer in module._modules.items():
                if isinstance(layer, nn.Linear):
                    last_linear = name
                if isinstance(layer, nn.ReLU) and last_linear:
                    fuse_list.append([last_linear, name])
                    last_linear = None
            if fuse_list:
                quant.fuse_modules(module, fuse_list, inplace=True)
    #model.qconfig = quant.get_default_qconfig('fbgemm')

    backend = "qnnpack"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model.cpu()
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

    return model_static_quantized

    # qconfig = QConfig(
    #     activation=MinMaxObserver.with_args(reduce_range=True),
    #     weight=MinMaxObserver.with_args(dtype=torch.qint8, reduce_range=True)
    # )
    # model.qconfig = qconfig

    # quant.prepare(model, inplace=True)
    # with torch.no_grad():
    #     for images, labels in data_loader_calib:
    #         images, labels = images.to(device), labels.to(device)
    #         model(images)
    #         break
    # quant.convert(model, inplace=True) # RuntimeError: Unsupported qscheme: per_channel_affine
    # return model

model_cifar_quant   = quantize_model(model_cifar, test_loader_cifar, device)
model_inette_quant  = quantize_model(model_inette, test_loader_inette, device)
model_iwoof_quant   = quantize_model(model_iwoof, test_loader_iwoof, device)

def evaluate_model(model, data_loader, device=torch.device('cpu')):
    model.to(device)
    model.eval()
    start_time = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    elapsed = time.time() - start_time
    accuracy = 100.0 * correct / total
    throughput = total / elapsed
    return accuracy, throughput, elapsed

acc_fp32_cifar, speed_fp32_cifar, time_fp32_cifar = evaluate_model(model_cifar, test_loader_cifar, device)
acc_int8_cifar, speed_int8_cifar, time_int8_cifar = evaluate_model(model_cifar_quant, test_loader_cifar, device)
acc_fp32_inette, speed_fp32_inette, _ = evaluate_model(model_inette, test_loader_inette, device)
acc_int8_inette, speed_int8_inette, _ = evaluate_model(model_inette_quant, test_loader_inette, device)
acc_fp32_iwoof, speed_fp32_iwoof, _ = evaluate_model(model_iwoof, test_loader_iwoof, device)
acc_int8_iwoof, speed_int8_iwoof, _ = evaluate_model(model_iwoof_quant, test_loader_iwoof, device)

print(f"CIFAR-10 FP32 Accuracy: {acc_fp32_cifar:.2f}%, Quantized Accuracy: {acc_int8_cifar:.2f}%")
print(f"ImageNette FP32 Accuracy: {acc_fp32_inette:.2f}%, Quantized Accuracy: {acc_int8_inette:.2f}%")
print(f"ImageWoof FP32 Accuracy: {acc_fp32_iwoof:.2f}%, Quantized Accuracy: {acc_int8_iwoof:.2f}%")
print(f"CIFAR-10 FP32 Throughput: {speed_fp32_cifar:.2f} img/s, Quantized: {speed_int8_cifar:.2f} img/s")
print(f"ImageNette FP32 Throughput: {speed_fp32_inette:.2f} img/s, Quantized: {speed_int8_inette:.2f} img/s")
print(f"ImageWoof FP32 Throughput: {speed_fp32_iwoof:.2f} img/s, Quantized: {speed_int8_iwoof:.2f} img/s")

torch.save(model_cifar.state_dict(), "vgg16_cifar_fp32.pth")
torch.save(model_cifar_quant.state_dict(), "vgg16_cifar_int8.pth")
fp32_size = os.path.getsize("vgg16_cifar_fp32.pth") / (1024*1024)
int8_size = os.path.getsize("vgg16_cifar_int8.pth") / (1024*1024)
print(f"FP32 model size = {fp32_size:.1f} MB, INT8 model size = {int8_size:.1f} MB")