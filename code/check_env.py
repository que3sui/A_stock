"""Environment sanity check."""
import sys

print("Python:", sys.version.split()[0])

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"matmul on GPU: {y.shape} OK, allocated {torch.cuda.memory_allocated()/1e6:.1f}MB")

import pandas, numpy, pyarrow, sklearn, lightgbm, matplotlib, jinja2, tqdm
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("pyarrow:", pyarrow.__version__)
print("sklearn:", sklearn.__version__)
print("lightgbm:", lightgbm.__version__)
print("ALL OK")
