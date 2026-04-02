import torch
from safetensors.torch import load_file
import os

path = "data/models/personalized_models/FurkanV1_Clean"
sf_file = os.path.join(path, "adapter_model.safetensors")
bin_file = os.path.join(path, "adapter_model.bin")

print(f"Converting {sf_file} to {bin_file}...")
weights = load_file(sf_file)
torch.save(weights, bin_file)
print("Done.")
