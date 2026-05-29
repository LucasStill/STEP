import subprocess
import itertools

# Define parameter lists
datasets = ['FD001', 'FD002', 'FD003', 'FD004']
latent_dims = [4, 8, 16, 32]
alphas = [0.5, 1.0, 2.0]
betas = [1.0, 0.5, 2.0]
epochs = 500

# Generate all combinations
combinations = list(itertools.product(datasets, latent_dims, alphas, betas))

print(f"Total number of commands to execute: {len(combinations)}\n")

# Execute each command sequentially
for i, (dataset, latent_dim, alpha, beta) in enumerate(combinations, 1):
    cmd = [
        'python', 'CMAPSS/step_cmapss.py',
        '--dataset', dataset,
        '--latent_dim', str(latent_dim),
        '--alpha', str(alpha),
        '--beta', str(beta),
        '--epochs', str(epochs),
        '--no_plotting'
    ]

    print(f"[{i}/{len(combinations)}] Running: dataset={dataset}, latent_dim={latent_dim}, alpha={alpha}, beta={beta}")

    try:
        subprocess.run(cmd, check=True)
        print(f"[{i}/{len(combinations)}] ✓ Completed\n")
    except subprocess.CalledProcessError as e:
        print(f"[{i}/{len(combinations)}] ✗ Failed with return code {e.returncode}\n")
    except Exception as e:
        print(f"[{i}/{len(combinations)}] ✗ Error: {e}\n")

print("All experiments finished!")
