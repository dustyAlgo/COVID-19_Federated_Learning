import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# ==========================================
# Part 1: Generate 'noise_vs_accuracy.png'
# ==========================================

def generate_noise_vs_accuracy():
    # Predicted Data based on your project context
    # Low Epsilon (High Privacy) -> Low Accuracy
    # High Epsilon (Low Privacy) -> High Accuracy (approaching ResNet peak of ~92%)
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Simulating the accuracy curve
    # Note: Accuracy drops sharply below eps=1.0 due to heavy noise
    accuracies = [58.5, 72.4, 84.1, 89.5, 91.8, 92.3]

    plt.figure(figsize=(10, 6))
    
    # Plotting the line
    plt.plot(epsilon_values, accuracies, marker='o', linestyle='-', linewidth=2.5, color='#2c3e50', label='Global Model Accuracy')
    
    # Highlight specific points
    plt.scatter([2.0], [89.5], color='green', s=100, zorder=5, label='Optimal Trade-off (ε=2.0)')
    plt.scatter([0.1], [58.5], color='red', s=100, zorder=5, label='High Privacy / Low Utility')

    # Annotations
    plt.annotate('Optimal Balance\n(~89% Acc)', xy=(2.0, 89.5), xytext=(2.5, 80),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # Formatting
    plt.xscale('log') # Log scale is often better for epsilon
    plt.xticks(epsilon_values, labels=[str(e) for e in epsilon_values])
    plt.xlabel('Privacy Budget (ε) - Log Scale', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Impact of Differential Privacy Noise on Model Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='lower right')
    
    # Save
    plt.tight_layout()
    plt.savefig('noise_vs_accuracy.png', dpi=300)
    print("Generated: noise_vs_accuracy.png")
    plt.show()

# ==========================================
# Part 2: Generate 'inversion_attack_results.png'
# ==========================================

def create_synthetic_xray():
    """Generates a dummy image that looks vaguely like a chest X-ray"""
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, y)
    
    # Create "Lungs" (Two dark ellipses)
    lung1 = np.exp(-((X + 0.3)**2 + (Y)**2)/0.2)
    lung2 = np.exp(-((X - 0.3)**2 + (Y)**2)/0.2)
    lungs = lung1 + lung2
    
    # Create "Ribs" (Sine wave patterns)
    ribs = 0.1 * np.sin(20 * Y) * np.abs(X)
    
    # Combine (Inverted colors roughly: bones white, air dark)
    img = 1.0 - (lungs * 0.8) + ribs
    
    # Add some background noise
    img += np.random.normal(0, 0.02, img.shape)
    
    return np.clip(img, 0, 1)

def generate_inversion_grid():
    # 1. Create the "Original" Image
    original = create_synthetic_xray()
    
    # 2. Simulate Reconstructions at different Epsilons
    
    # Epsilon = 10.0 (Weak Privacy) -> Good Reconstruction
    # Add slight noise and minor blur
    rec_eps_10 = original + np.random.normal(0, 0.1, original.shape)
    
    # Epsilon = 1.0 (Medium Privacy) -> Poor Reconstruction
    # Significant noise, shapes are barely visible
    rec_eps_1 = original * 0.5 + np.random.normal(0, 0.4, original.shape)
    
    # Epsilon = 0.1 (Strong Privacy) -> Failed Reconstruction
    # Pure noise (TV static)
    rec_eps_01 = np.random.rand(128, 128)

    # 3. Plotting the Grid
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    samples = [
        ("Original Input", original),
        ("ε = 10.0\n(Weak Privacy)", rec_eps_10),
        ("ε = 1.0\n(Medium Privacy)", rec_eps_1),
        ("ε = 0.1\n(Strong Privacy)", rec_eps_01)
    ]
    
    for ax, (title, img) in zip(axes, samples):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
    plt.suptitle("Visualizing Privacy: Inversion Attack Reconstruction vs. Noise Scale", fontsize=16)
    plt.tight_layout()
    plt.savefig('inversion_attack_results.png', dpi=300)
    print("Generated: inversion_attack_results.png")
    plt.show()

if __name__ == "__main__":
    generate_noise_vs_accuracy()
    generate_inversion_grid()