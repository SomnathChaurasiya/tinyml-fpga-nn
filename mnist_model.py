import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Device
device = torch.device("cpu")

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten to 784
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

# Simple MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training (short)
for epoch in range(3):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training Done")

# -------------------------------
# Quantization
# -------------------------------

def quantize_tensor(tensor, scale):
    return torch.clamp((tensor / scale).round(), -128, 127).to(torch.int8)

scale_w = 0.02
scale_x = 0.02

fc1_w = quantize_tensor(model.fc1.weight.data, scale_w)
fc1_b = (model.fc1.bias.data / (scale_w * scale_x)).round().to(torch.int32)

fc2_w = quantize_tensor(model.fc2.weight.data, scale_w)
fc2_b = (model.fc2.bias.data / (scale_w * scale_x)).round().to(torch.int32)

# -------------------------------
# Export files
# -------------------------------

np.savetxt("fc1_weights.txt", fc1_w.numpy(), fmt="%d")
np.savetxt("fc1_bias.txt", fc1_b.numpy(), fmt="%d")

np.savetxt("fc2_weights.txt", fc2_w.numpy(), fmt="%d")
np.savetxt("fc2_bias.txt", fc2_b.numpy(), fmt="%d")

# Export one test sample
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

sample, label = test_dataset[0]

input_q = quantize_tensor(sample, scale_x)

np.savetxt("input.txt", input_q.numpy(), fmt="%d")

print("Export Done")
