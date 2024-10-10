# Quantum Entanglement Trees: Optimizing Quantized Matrix Quantization Via Element Replacement And Residual Clustering (ICLR2025 Under Review)

## Introduction

This project provides Python implementations of five quantization methods for efficient matrix compression:

- **QET_Vanilla**: Basic Version of Quantum Entanglement Trees ( **Our Method** )
- **QET**:Advanced Version of Quantum Entanglement Trees ( **Our Method** )
- **LOPQ**: Locally Optimized Product Quantization
- **OPQ**: Optimized Product Quantization
- **PQ**: Product Quantization

Our primary contribution is the **QET_Vanilla** and **QET** method, which are designed to improve upon existing PQ-based quantization techniques by element replacement and residual clustering.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Algorithms](#algorithms)
  - [QET_Vanilla (Our Method Basic Version)](#qet_vanialla-our-method)
  - [QET (Our Method Advanced Version)](#qet_vanialla-our-method)
  - [LOPQ](#lopq)
  - [OPQ](#opq)
  - [PQ](#pq)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [QET Example](#qet-example)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- **High Performance**: Optimized for fast and scalable matrix compression.
- **Customizable**: Flexible parameter tuning (e.g., subspaces, codewords, compression rates) to meet different requirements.
- **Modular**: Easy integration into existing workflows and systems.
- **Educational**: Clean and well-documented code for learning and further development.

## Algorithms

### QET_Vanilla ( Basic Version of Quantum Entanglement Trees )

**QET_Vanilla** introduces element replacement to traditional product quantization, improving the efficiency of encoding and decoding high-dimensional data.


### QET ( Advanced Version of Quantum Entanglement Trees )

**QET** builds on QET_Vanilla by incorporating Residual Quantization Optimization (RQO) and Codebook Quantization Optimization (CQO), achieving significantly improved accuracy with an acceptable computational cost.



### LOPQ

**Locally Optimized Product Quantization (LOPQ)** enhances standard PQ by optimizing quantization locally within clusters, leading to improved accuracy in nearest neighbor search tasks.

### OPQ

**Optimized Product Quantization (OPQ)** introduces a rotation matrix to minimize quantization errors, making it highly effective for approximate nearest neighbor searches in high-dimensional spaces.

### PQ

**Product Quantization (PQ)** is a fundamental method for compressing high-dimensional vectors. It divides vectors into subspaces and quantizes each subspace separately, enabling fast and memory-efficient approximate nearest neighbor searches.

## Requirements

- Python 3.x
- NumPy
- SciPy

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/QET-release/QET-release.git
   ```

2. **Navigate to the project directory**

   ```bash
   cd QET-release
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure that `requirements.txt` contains:

   ```
   numpy
   scipy
   ```

## Usage

### QET Example

Here's how to use our **QET** implementation:

```python
import numpy as np
from QET import QET

# Generate random data
data = np.random.rand(10000, 128)  # 10000 samples, each with 128 dimensions

# Initialize the QET model with the desired parameters
qet = QET(
    M = 4,                 # Number of subspaces for dividing the data
    MemorySize = 256,      # Target memory usage after compression in bytes
)

# Train the QET model on the data
qet.fit(data)

# Encode the data using the QET model
code, imap = qet.encode(data)

# Decode the data back to its original form
decoded_data = qet.decode(code, imap)
  
# Compute and display reconstruction error
error = np.linalg.norm(data - decoded_data)
print(f"Reconstruction Error: {error}")

```

## Project Structure

```plaintext

├── QET_Vanilla.py      # Basic QET implementation (element replacement only)
├── QET.py              # Advanced QET implementation (includes RQO and CQO)
├── lopq.py             # LOPQ implementation
├── opq.py              # OPQ implementation
├── pq.py               # Standard PQ implementation
├── requirements.txt    # Project dependencies
├── README.md           # Documentation and project description

```

## Contributing

Contributions are welcome! You can contribute in the following ways:

- **Reporting Issues**: Submit issues for bugs or feature requests.
- **Pull Requests**: Fork the repository and submit pull requests for improvements.
- **Documentation**: Help improve the documentation.

Please ensure your contributions adhere to the project's coding standards and styles.

## License

This project is licensed under the [MIT License](LICENSE).

- **Note**: Some code in this project is adapted from external sources. Specifically:
  - `pq.py` and portions of `opq.py` are adapted from the [nanopq](https://github.com/matsui528/nanopq) library by **Yusuke Matsui**.
  - If you use these portions of the code, please adhere to the licensing terms specified in the original repository.

## Acknowledgments

- **Yusuke Matsui** for the nanopq library, which inspired parts of this project.
- The community for continuous support and contributions.

---

If you have any questions or need further assistance, feel free to open an issue or contact the maintainers.
