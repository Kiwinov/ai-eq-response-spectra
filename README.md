# Conditional GAN for Earthquake Response Spectrum Generation

This project implements a **Conditional Generative Adversarial Network (cGAN)** to generate earthquake acceleration data in the log-frequency domain (**response spectra**).  
The model is trained using a **novel composite loss function** combining:

- **Binary Cross-Entropy (BCE)** — for adversarial training.
- **Smoothness Regularization** — to ensure realistic log-frequency-domain curves.
- **Mean Squared Error (MSE)** — to match ground truth response spectra.

On the entire test set, the model achieved an **R² score of 0.9121**, demonstrating strong predictive capability.

---

## Project Structure

```

.
├── data/         # Dataset files (response spectra: period vs. log(acceleration))
├── models/       # Saved model weights and checkpoints
├── src/          # Source code for training, evaluation, and generation
├── pyproject.toml
├── uv.lock
└── README.md

````

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast Python dependency management.  
You must have **Python 3.9+** installed.

1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

Or, if using Windows (PowerShell):

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

2. Clone the repository

```bash
git clone https://github.com/yourusername/earthquake-cgan.git
cd earthquake-cgan
```

3. Install dependencies

```bash
uv sync
```

---

## Training

1. Place your earthquake response spectrum data in the `data/` folder. Please refer to the alraedy existing `.npy` files.
2. Run the training via the Jupyter Notebook:

```bash
uv run python src/CGAN.ipynb
```

3. Checkpoints will be saved in the `models/intermediate` directory.

---

## Evaluation

The evaluation script is at the end of `src/CGAN.ipynb` file.


## Novel Loss Function

The total loss function used in training is:

```
Loss = BCE_GAN + λ₂ * SmoothnessLoss + λ₃ * MSE
```

Where:

* `BCE_GAN` — binary cross-entropy for adversarial training.
* `SmoothnessLoss` — penalizes large second derivatives to encourage smooth spectral curves.
* `MSE` — enforces similarity to ground truth spectra.

