import torch

# Set CUDA settings for better performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.cuda.empty_cache()  # Clear GPU cache before training