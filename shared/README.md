# Shared Utilities

This directory contains utilities and helper functions that can be used across multiple projects.

## Contents

- `data_utils`: Common data loading, processing, and augmentation utilities
- `visualization`: Reusable visualization components and plotting functions
- `evaluation`: Model evaluation metrics and reporting tools

## Usage

Import these utilities in your project code or notebooks:

```python
import sys
import os

# Add the root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from shared modules
from shared.data_utils import data_loader
from shared.visualization import plot_utils
from shared.evaluation import metrics
```