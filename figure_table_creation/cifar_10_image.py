import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random

transform = transforms.ToTensor()
cifar10 = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

index = random.randint(0, len(cifar10) - 1)
image, label = cifar10[index]

image_np = image.permute(1, 2, 0).numpy()

classes = cifar10.classes

plt.imshow(image_np)
plt.title(f"Label: {classes[label]}")
plt.axis('off')
plt.show()
