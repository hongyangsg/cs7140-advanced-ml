import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math


class TwoLayerNN(nn.Module):
    """Two-layer neural network classifier for MNIST"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten images
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def relu_ntk_kernel(X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    XZ = X @ Z.T                                  # (n, m)

    # Normalise for angle computation
    norm_X = torch.norm(X, dim=1, keepdim=True)   # (n, 1)
    norm_Z = torch.norm(Z, dim=1, keepdim=True)   # (m, 1)
    cos_theta = torch.clamp(XZ / (norm_X * norm_Z.T + 1e-10), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)                  # (n, m)

    # Arc-cosine kernel order 0: E[relu'(w^T x) relu'(w^T z)] = (pi - theta) / (2*pi)
    # Second layer is fixed, so k1 (from d/da_r) drops out.
    # Kernel reduces to: K_NTK(x, z) = (x^T z) * k0(x, z)
    k0 = (math.pi - theta) / (2 * math.pi)

    return XZ * k0

class NTKClassifier:
    def __init__(self, lam: float = 1.0, device: torch.device = None):
        self.lam = lam
        self.device = device or torch.device('cpu')
        self.alpha = None
        self.X_train = None

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, n_classes: int = 10):
        n = X_train.shape[0]
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # One-hot encode labels
        Y = torch.zeros(n, n_classes, device=self.device)
        Y.scatter_(1, y_train.unsqueeze(1), 1.0)

        print(f"  Computing K_train ({n}x{n})...")
        K_train = relu_ntk_kernel(X_train, X_train)
        K_reg = K_train + self.lam * torch.eye(n, device=self.device)

        print(f"  Solving (K + {self.lam}*I) alpha = Y ...")
        self.alpha = torch.linalg.solve(K_reg, Y)   # (n, C)
        self.X_train = X_train

    def predict(self, X_test: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        X_test = X_test.to(self.device)
        scores = []
        for i in range(0, X_test.shape[0], batch_size):
            Xb = X_test[i : i + batch_size]
            scores.append(relu_ntk_kernel(Xb, self.X_train) @ self.alpha)
        return torch.argmax(torch.cat(scores, dim=0), dim=1)

    def score(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        preds = self.predict(X_test)
        return (preds == y_test.to(self.device)).float().mean().item()


def load_mnist_flat(n_train: int = 5000, n_test: int = 1000):
    """Load MNIST and return flat (N, 784) tensors (subset for kernel tractability)."""
    from torch.utils.data import Subset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_full  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def collect(dataset, n):
        loader = DataLoader(Subset(dataset, list(range(n))), batch_size=512, shuffle=False)
        Xs, ys = [], []
        for X, y in loader:
            Xs.append(X.view(X.size(0), -1))
            ys.append(y)
        return torch.cat(Xs), torch.cat(ys)

    return (*collect(train_full, n_train), *collect(test_full, n_test))


def load_mnist(batch_size=256):
    """Load and preprocess MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model, train_loader, test_loader, device, epochs=20, lr=0.001):
    """Train the neural network with mini-batch SGD"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        test_accuracies.append(accuracy)

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    return train_losses, test_accuracies


def measure_weights(model):
    """Extract and measure network weights"""
    B1 = math.sqrt(np.linalg.norm(model.fc1.weight.data.cpu().numpy()) ** 2
                   + np.linalg.norm(model.fc1.bias.data.cpu().numpy()) ** 2)
    B2 = math.sqrt(np.linalg.norm(model.fc2.weight.data.cpu().numpy()) ** 2
                   + np.linalg.norm(model.fc2.bias.data.cpu().numpy()) ** 2)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return B1, B2, num_params


def measure_path_norm(model):
    """Calculate the path norm of the two-layer nn"""
    path_norm = 0.0
    for i in range(model.hidden_dim):
        for k in range(model.output_dim):
            w1 = model.fc1.weight.data[i, :]
            b1 = model.fc1.bias.data[i]
            w2 = model.fc2.weight.data[k, i]
            b2 = model.fc2.bias.data[k]

            path_contribution = (torch.norm(w1, p=2) + torch.abs(b1)) * (torch.abs(w2) + torch.abs(b2))
            path_norm += path_contribution.item()

    return path_norm


def evaluate_full(model, data_loader, device):
    """Evaluate model on a full dataset, returning loss and accuracy"""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            num_batches += 1
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return total_loss / num_batches, correct / total


def main_two_layer_nn_mnist():
    # Load MNIST
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(batch_size=256)
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # Ablation study: vary hidden_dim
    input_dim = 28 * 28
    output_dim = 10
    hidden_dims = range(50, 51, 50)  # [50, 100, 150, ..., 500]

    results = []

    print("\n" + "="*80)
    print("ABLATION STUDY: Varying Hidden Dimension (2-Layer Network on MNIST)")
    print("="*80)

    for hidden_dim in hidden_dims:
        print(f"\n{'='*80}")
        print(f"Training model with hidden_dim = {hidden_dim}")
        print(f"{'='*80}")

        # Reset seed for fair comparison
        torch.manual_seed(42)

        # Initialize model
        model = TwoLayerNN(input_dim, hidden_dim, output_dim).to(device)

        # Train model
        train_losses, test_accuracies = train_model(
            model, train_loader, test_loader, device, epochs=20, lr=0.001)

        # Final evaluation
        train_loss, train_accuracy = evaluate_full(model, train_loader, device)
        test_loss, test_accuracy = evaluate_full(model, test_loader, device)

        # Measure weights
        B1, B2, num_params = measure_weights(model)
        agg_norm = B1 * B2 * math.sqrt(hidden_dim)

        path_norm = measure_path_norm(model)

        # Store results
        results.append({
            'hidden_dim': hidden_dim,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'B1': B1,
            'B2': B2,
            'num_params': num_params,
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
        })

        print(f"\nResults for hidden_dim = {hidden_dim} ({num_params} parameters):")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Accuracy:  {test_accuracy:.4f}")
        print(f"  Weight norms: B1={B1:.4f}, B2={B2:.4f}")
        print(f"  Aggregated norm: {agg_norm:.4f}")
        print(f"  Path norm: {path_norm:.4f}")

    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Hidden Dim':<12} {'# Params':<12} {'Train Loss':<12} {'Train Acc':<12} "
          f"{'Test Loss':<12} {'Test Acc':<12} {'B1':<10} {'B2':<10}")
    print("-"*90)
    for r in results:
        print(f"{r['hidden_dim']:<12} {r['num_params']:<12} {r['train_loss']:<12.4f} "
              f"{r['train_accuracy']:<12.4f} {r['test_loss']:<12.4f} {r['test_accuracy']:<12.4f} "
              f"{r['B1']:<10.4f} {r['B2']:<10.4f}")

    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    hidden_dims_list = [r['hidden_dim'] for r in results]
    train_losses_list = [r['train_loss'] for r in results]
    train_accs_list = [r['train_accuracy'] for r in results]
    test_losses_list = [r['test_loss'] for r in results]
    test_accs_list = [r['test_accuracy'] for r in results]

    ax1.plot(hidden_dims_list, train_losses_list, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Hidden Dimension')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Train Loss vs Hidden Dimension (2-Layer NN, MNIST)')
    ax1.grid(True)

    ax2.plot(hidden_dims_list, train_accs_list, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Hidden Dimension')
    ax2.set_ylabel('Train Accuracy')
    ax2.set_title('Train Accuracy vs Hidden Dimension (2-Layer NN, MNIST)')
    ax2.grid(True)

    ax3.plot(hidden_dims_list, test_losses_list, 'o-', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Hidden Dimension')
    ax3.set_ylabel('Test Loss')
    ax3.set_title('Test Loss vs Hidden Dimension (2-Layer NN, MNIST)')
    ax3.grid(True)

    ax4.plot(hidden_dims_list, test_accs_list, 'o-', linewidth=2, markersize=8, color='orange')
    ax4.set_xlabel('Hidden Dimension')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Test Accuracy vs Hidden Dimension (2-Layer NN, MNIST)')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('../figures/two_layer_nn_mnist_results.png', dpi=150)
    print("\nAblation study visualization saved to '../figures/two_layer_nn_mnist_results.png'")
    plt.show()


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── ReLU NTK Kernel Experiment ────────────────────────────────────────────
    print("\n" + "="*80)
    print("NTK KERNEL RIDGE REGRESSION (two-layer ReLU, infinite-width limit)")
    print("="*80)

    # Kernel methods are O(n^3) so we work on a manageable subset
    n_train_ntk, n_test_ntk = 10000, 1000
    print(f"\nLoading {n_train_ntk} train / {n_test_ntk} test samples for NTK experiment...")
    X_tr_ntk, y_tr_ntk, X_te_ntk, y_te_ntk = load_mnist_flat(n_train_ntk, n_test_ntk)
    print(f"  X_train: {X_tr_ntk.shape}, X_test: {X_te_ntk.shape}")

    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ntk_sweep = []
    for lam in lambdas:
        print(f"\nlambda = {lam}")
        clf = NTKClassifier(lam=lam, device=device)
        clf.fit(X_tr_ntk, y_tr_ntk)
        train_acc_ntk = clf.score(X_tr_ntk, y_tr_ntk)
        test_acc_ntk  = clf.score(X_te_ntk,  y_te_ntk)
        print(f"  Train Accuracy: {train_acc_ntk:.4f}  |  Test Accuracy: {test_acc_ntk:.4f}")
        ntk_sweep.append({'lam': lam, 'train_acc': train_acc_ntk, 'test_acc': test_acc_ntk})

    # Summary table
    print("\n" + "-"*50)
    print(f"{'lambda':<12} {'Train Acc':<12} {'Test Acc':<12}")
    print("-"*50)
    for r in ntk_sweep:
        print(f"{r['lam']:<12} {r['train_acc']:<12.4f} {r['test_acc']:<12.4f}")

    # Plot NTK lambda sweep
    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx([r['lam'] for r in ntk_sweep], [r['train_acc'] for r in ntk_sweep],
                'o-', linewidth=2, markersize=8, label='Train Accuracy')
    ax.semilogx([r['lam'] for r in ntk_sweep], [r['test_acc'] for r in ntk_sweep],
                's--', linewidth=2, markersize=8, label='Test Accuracy')
    ax.set_xlabel('Regularisation lambda')
    ax.set_ylabel('Accuracy')
    ax.set_title('ReLU NTK Kernel Ridge Regression\nTwo-layer NN (MNIST subset)')
    ax.legend()
    ax.grid(True, which='both')
    plt.tight_layout()
    plt.savefig('../figures/two_layer_ntk_mnist.png', dpi=150, bbox_inches='tight')
    print("\nNTK figure saved to '../figures/two_layer_ntk_mnist.png'")
    plt.show()

    return ntk_sweep


if __name__ == "__main__":
    ntk_sweep = main()
