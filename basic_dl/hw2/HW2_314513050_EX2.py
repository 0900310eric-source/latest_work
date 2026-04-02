import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import math

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

# preprocessing for CIFAR-10
mean = [0.4914, 0.4822, 0.4465]
std  = [0.2470, 0.2435, 0.2616]

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# hyperparameters
lr = 5e-4
epochs = 15
batch_size = 128
l2_alpha = 0      # L2 strength

# kernel / stride config (for experiments)
FILTER_CONFIG = {
    "k1": 7, "k2": 7, "k3": 7,
    "s1": 1, "s2": 1, "s3": 1,
}


class CIFAR10CNN(nn.Module):
    def __init__(self, k1=7, k2=7, k3=7, s1=1, s2=1, s3=1):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=k1, stride=s1, padding=k1 // 2)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=k2, stride=s2, padding=k2 // 2)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=k3, stride=s3, padding=k3 // 2)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feat = self._forward_features(dummy)
            self.flatten_dim = feat.view(1, -1).shape[1]

        self.fc1   = nn.Linear(self.flatten_dim, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2   = nn.Linear(256, 10)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def _forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu(self.bn2(self.conv2(x))))
        x = self.pool2(self.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.bn_fc(self.fc1(x))))
        x = self.fc2(x)
        return x


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            loss = criterion(outs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def show_examples(model, loader, num_correct=3, num_wrong=3):
    model.eval()
    correct_imgs, wrong_imgs = [], []
    classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    mean_t = torch.tensor(mean)
    std_t  = torch.tensor(std)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            for img, label, pred in zip(imgs, labels, preds):
                li = int(label.item())
                pi = int(pred.item())
                if li == pi and len(correct_imgs) < num_correct:
                    correct_imgs.append((img.cpu(), li, pi))
                elif li != pi and len(wrong_imgs) < num_wrong:
                    wrong_imgs.append((img.cpu(), li, pi))
                if len(correct_imgs) >= num_correct and len(wrong_imgs) >= num_wrong:
                    break
            if len(correct_imgs) >= num_correct and len(wrong_imgs) >= num_wrong:
                break

    imgs_to_show = wrong_imgs + correct_imgs
    total = len(imgs_to_show)

    plt.figure(figsize=(2 * total, 3))
    for i, (img, label, pred) in enumerate(imgs_to_show):
        plt.subplot(1, total, i + 1)
        img_show = img.permute(1, 2, 0) * std_t + mean_t
        img_show = torch.clamp(img_show, 0, 1)
        plt.imshow(img_show)
        plt.axis("off")
        color = "red" if label != pred else "lime"
        plt.title(f"T:{classes[label]}\nP:{classes[pred]}", color=color, fontsize=9)
    plt.suptitle("Correct (green) / Wrong (red)", y=1.05)
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(model, loader):
    model.eval()
    imgs, labels = next(iter(loader))
    img = imgs[0].unsqueeze(0).to(device)
    label = int(labels[0].item())
    mean_t = torch.tensor(mean)
    std_t  = torch.tensor(std)

    with torch.no_grad():
        out1 = model.relu(model.bn1(model.conv1(img)))
        out2 = model.pool1(model.relu(model.bn2(model.conv2(out1))))
        out3 = model.pool2(model.relu(model.bn3(model.conv3(out2))))
        pred = int(torch.argmax(model(img)).item())

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    img_show = imgs[0].permute(1, 2, 0) * std_t + mean_t
    img_show = torch.clamp(img_show, 0, 1)
    plt.imshow(img_show)
    plt.axis("off")
    plt.title(f"Label:{label}, Pred:{pred}")

    for i in range(3):
        plt.subplot(1, 4, i + 2)
        plt.imshow(out1[0, i].cpu(), cmap="gray")
        plt.axis("off")
        plt.title(f"Conv1 ch{i}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 4, i + 1)
        plt.imshow(out2[0, i].cpu(), cmap="gray")
        plt.axis("off")
        plt.title(f"Conv2 ch{i}")
    plt.subplot(1, 4, 4)
    plt.imshow(out3[0, 0].cpu(), cmap="gray")
    plt.axis("off")
    plt.title("Conv3 ch0")
    plt.tight_layout()
    plt.show()


def plot_weight_histograms_all_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))

    weight_data = []
    for name, module in layers:
        w = module.weight.detach().cpu().numpy().ravel()
        weight_data.append((f"{name}.weight", w))

    n = len(weight_data)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, data) in zip(axes, weight_data):
        ax.hist(data, bins=100)
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number")

    for k in range(len(weight_data), len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()


def plot_bias_histograms_all_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            layers.append((name, module))

    bias_data = []
    for name, module in layers:
        b = module.bias.detach().cpu().numpy().ravel()
        bias_data.append((f"{name}.bias", b))

    n = len(bias_data)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, data) in zip(axes, bias_data):
        ax.hist(data, bins=100)
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number")

    for k in range(len(bias_data), len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()


def main():
    print("Training CIFAR-10 CNN...")

    full_train = datasets.CIFAR10(root="./", train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root="./", train=False, download=True, transform=transform_test)

    train_size, val_size = 45000, 5000
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=4)

    model = CIFAR10CNN(**FILTER_CONFIG).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    train_losses, val_losses = [], []
    train_accs, val_accs, test_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            ce_loss = criterion(outs, labels)

            l2_norm = 0.0
            if l2_alpha > 0:
                for name, p in model.named_parameters():
                    if p.requires_grad and ("weight" in name):
                        l2_norm += torch.sum(p ** 2)
            loss = ce_loss + 0.5 * l2_alpha * l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        val_loss = evaluate_loss(model, val_loader, criterion)
        val_acc  = evaluate_acc(model, val_loader)
        test_acc = evaluate_acc(model, test_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | "
            f"Test Acc: {test_acc*100:.2f}%"
        )

    print("Training finished!")

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs_range, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.xticks(np.arange(1, epochs + 1, 1))

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label="Train Acc", marker="o")
    plt.plot(epochs_range, val_accs, label="Val Acc", marker="o")
    plt.plot(epochs_range, test_accs, label="Test Acc", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.xticks(np.arange(1, epochs + 1, 1))
    plt.tight_layout()
    plt.show()

    print("Plotting weight histograms for all layers...")
    plot_weight_histograms_all_layers(model)

    print("Plotting bias histograms for all layers...")
    plot_bias_histograms_all_layers(model)

    print("Showing correct / wrong examples...")
    show_examples(model, test_loader)

    print("Visualizing feature maps...")
    visualize_feature_maps(model, test_loader)


if __name__ == "__main__":
    main()
