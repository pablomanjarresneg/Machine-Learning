import sys
print("Current Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying to import shared.data_utils:")
try:
    from shared.data_utils import load_flux_from_fits, normalize_flux
    print("Success!")
except ImportError as e:
    print(f"Error: {e}")

print("\nTrying with explicit sys.path addition:")
import os
project_root = os.path.abspath(".")
print(f"Adding {project_root} to sys.path")
sys.path.append(project_root)
try:
    from shared.data_utils import load_flux_from_fits, normalize_flux
    print("Success after path addition!")
except ImportError as e:
    print(f"Error after path addition: {e}")