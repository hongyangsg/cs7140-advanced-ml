import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math


class ThreeLayerNN(nn.Module):
    """Three-layer neural network classifier"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def generate_synthetic_data(n_samples=1000, dataset_type='moons'):
    """Generate synthetic dataset for classification"""
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif dataset_type == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   random_state=42)
    else:
        raise ValueError("dataset_type must be 'moons' or 'classification'")

    return X, y


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
    """Train the neural network"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test).float().mean()
            test_accuracies.append(accuracy.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

    return train_losses, test_accuracies


def measure_weights(model):
    """Extract and measure network weights"""
    print("\n" + "="*60)
    print("WEIGHT MEASUREMENTS")
    print("="*60)

    B1 = math.sqrt(np.linalg.norm(model.fc1.weight.data.cpu().numpy()) ** 2 + np.linalg.norm(model.fc1.bias.data.cpu().numpy()) ** 2)
    B2 = math.sqrt(np.linalg.norm(model.fc2.weight.data.cpu().numpy()) ** 2 + np.linalg.norm(model.fc2.bias.data.cpu().numpy()) ** 2)
    B3 = math.sqrt(np.linalg.norm(model.fc3.weight.data.cpu().numpy()) ** 2 + np.linalg.norm(model.fc3.bias.data.cpu().numpy()) ** 2)

    # calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return B1, B2, B3, num_params


def measure_path_norm(model):
    """Calculate the path norm of the three-layer nn"""
    path_norm = 0.0
    for i in range(model.hidden_dim):  # first hidden layer neurons
        for j in range(model.hidden_dim):  # second hidden layer neurons
            for k in range(model.output_dim):  # output layer neurons
                w1 = model.fc1.weight.data[i, :]
                b1 = model.fc1.bias.data[i]
                w2 = model.fc2.weight.data[j, i]
                b2 = model.fc2.bias.data[j]
                w3 = model.fc3.weight.data[k, j]
                b3 = model.fc3.bias.data[k]

                path_contribution = (torch.norm(w1, p=2) + torch.abs(b1)) * (torch.abs(w2) + torch.abs(b2)) * (torch.abs(w3) + torch.abs(b3))
                path_norm += path_contribution.item()

    return path_norm


def visualize_decision_boundary(model, X, y, title="Decision Boundary"):
    """Visualize the decision boundary of the trained model"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    model.eval()
    with torch.no_grad():
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = torch.argmax(Z, dim=1)
        Z = Z.reshape(xx.shape).numpy()

    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')


def plot_training_history(train_losses, test_accuracies):
    """Plot training loss and test accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.plot(test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.tight_layout()


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate synthetic data
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_data(n_samples=1000, dataset_type='moons')

    # Split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Ablation study: vary hidden_dim from 10 to 100
    input_dim = 2
    output_dim = 2
    hidden_dims = range(10, 101, 10)  # [10, 20, 30, ..., 100]

    results = []

    print("\n" + "="*80)
    print("ABLATION STUDY: Varying Hidden Dimension (3-Layer Network)")
    print("="*80)

    for hidden_dim in hidden_dims:
        print(f"\n{'='*80}")
        print(f"Training model with hidden_dim = {hidden_dim}")
        print(f"{'='*80}")

        # Initialize model
        model = ThreeLayerNN(input_dim, hidden_dim, output_dim)

        # Train model
        train_losses, test_accuracies = train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            train_loss = nn.CrossEntropyLoss()(train_outputs, y_train)
            _, train_predicted = torch.max(train_outputs, 1)
            train_accuracy = (train_predicted == y_train).float().mean()

            test_outputs = model(X_test)
            test_loss = nn.CrossEntropyLoss()(test_outputs, y_test)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = (test_predicted == y_test).float().mean()

        # Store results
        results.append({
            'hidden_dim': hidden_dim,
            'train_loss': train_loss.item(),
            'train_accuracy': train_accuracy.item(),
            'test_loss': test_loss.item(),
            'test_accuracy': test_accuracy.item()
        })

        print(f"\nResults for hidden_dim = {hidden_dim}:")
        print(f"  Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_accuracy.item():.4f}")
        print(f"  Test Loss:  {test_loss.item():.4f}, Test Accuracy:  {test_accuracy.item():.4f}")

    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Hidden Dim':<12} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")
    print("-"*80)
    for result in results:
        print(f"{result['hidden_dim']:<12} {result['train_loss']:<12.4f} {result['train_accuracy']:<12.4f} "
              f"{result['test_loss']:<12.4f} {result['test_accuracy']:<12.4f}")

    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    hidden_dims_list = [r['hidden_dim'] for r in results]
    train_losses_list = [r['train_loss'] for r in results]
    train_accs_list = [r['train_accuracy'] for r in results]
    test_losses_list = [r['test_loss'] for r in results]
    test_accs_list = [r['test_accuracy'] for r in results]

    # Plot train loss vs hidden_dim
    ax1.plot(hidden_dims_list, train_losses_list, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Hidden Dimension')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Train Loss vs Hidden Dimension (3-Layer NN)')
    ax1.grid(True)

    # Plot train accuracy vs hidden_dim
    ax2.plot(hidden_dims_list, train_accs_list, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Hidden Dimension')
    ax2.set_ylabel('Train Accuracy')
    ax2.set_title('Train Accuracy vs Hidden Dimension (3-Layer NN)')
    ax2.grid(True)

    # Plot test loss vs hidden_dim
    ax3.plot(hidden_dims_list, test_losses_list, 'o-', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Hidden Dimension')
    ax3.set_ylabel('Test Loss')
    ax3.set_title('Test Loss vs Hidden Dimension (3-Layer NN)')
    ax3.grid(True)

    # Plot test accuracy vs hidden_dim
    ax4.plot(hidden_dims_list, test_accs_list, 'o-', linewidth=2, markersize=8, color='orange')
    ax4.set_xlabel('Hidden Dimension')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Test Accuracy vs Hidden Dimension (3-Layer NN)')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('../figures/ablation_study_hidden_dim.png', dpi=150)
    print("\nAblation study visualization saved to '../figures/ablation_study_hidden_dim.png'")
    plt.show()

    return results


if __name__ == "__main__":
    results = main()
