import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import math

# basic settings
transform = transforms.ToTensor()
lr = 5e-4
epochs = 10
batch_size = 64
l2_alpha = 5e-2           # set 0 for "no L2" run

# kernel / stride config (for experiments)
K1, K2 = 7, 7
S1, S2 = 1, 1

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")


class MNISTCNN(nn.Module):
    def __init__(self, k1=7, k2=7, s1=1, s2=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, k1, stride=s1, padding=k1 // 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, k1, stride=s1, padding=k1 // 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, k2, stride=s2, padding=k2 // 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, k2, stride=s2, padding=k2 // 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            feat = self.block2(self.block1(dummy))
            flatten_dim = feat.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
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


def show_classification_examples(model, loader, num_correct=3, num_wrong=3):
    model.eval()
    correct_imgs, wrong_imgs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            for img, label, pred in zip(imgs, labels, preds):
                label_i = int(label.item())
                pred_i = int(pred.item())
                if label_i == pred_i and len(correct_imgs) < num_correct:
                    correct_imgs.append((img.cpu(), label_i, pred_i))
                elif label_i != pred_i and len(wrong_imgs) < num_wrong:
                    wrong_imgs.append((img.cpu(), label_i, pred_i))
                if len(correct_imgs) >= num_correct and len(wrong_imgs) >= num_wrong:
                    break
            if len(correct_imgs) >= num_correct and len(wrong_imgs) >= num_wrong:
                break

    imgs_to_show = wrong_imgs + correct_imgs
    total = len(imgs_to_show)
    plt.figure(figsize=(2 * total, 3))
    for i, (img, label, pred) in enumerate(imgs_to_show):
        plt.subplot(1, total, i + 1)
        plt.imshow(img.squeeze().cpu(), cmap="gray")
        plt.axis("off")
        color = "red" if label != pred else "lime"
        plt.title(f"label:{label}, pred:{pred}", fontsize=9, pad=5, color=color)
    plt.suptitle("Correct (green) / Wrong (red)", fontsize=12, y=1.05)
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(model, loader):
    model.eval()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)
    img = imgs[0].unsqueeze(0)
    label = int(labels[0].item())
    with torch.no_grad():
        out_block1 = model.block1(img)
        out_block2 = model.block2(out_block1)
        pred = int(torch.argmax(model(img)).item())

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img.squeeze().cpu(), cmap="gray")
    plt.title(f"label:{label}, pred:{pred}")
    plt.axis("off")
    for i in range(3):
        plt.subplot(1, 4, i + 2)
        plt.imshow(out_block1[0, i].cpu(), cmap="gray")
        plt.title(f"block1 ch{i}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img.squeeze().cpu(), cmap="gray")
    plt.title(f"label:{label}, pred:{pred}")
    plt.axis("off")
    for i in range(3):
        plt.subplot(1, 4, i + 2)
        plt.imshow(out_block2[0, i].cpu(), cmap="gray")
        plt.title(f"block2 ch{i}")
        plt.axis("off")
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
    print("Starting MNIST training...")

    full_train = datasets.MNIST(root="./", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./", train=False, download=True, transform=transform)
    train_size, val_size = 55000, 5000
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(full_train, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = MNISTCNN(k1=K1, k2=K2, s1=S1, s2=S2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    train_losses, val_losses, train_accs, val_accs, test_accs = [], [], [], [], []

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
        val_acc = evaluate_acc(model, val_loader)
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

    show_classification_examples(model, test_loader, num_correct=3, num_wrong=3)
    visualize_feature_maps(model, test_loader)


if __name__ == "__main__":
    main()
