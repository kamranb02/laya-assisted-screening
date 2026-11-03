
import torch
print('PyTorch Version:', torch.__version__)
cuda_available = torch.cuda.is_available()
print('CUDA Available:', cuda_available)
if cuda_available:
    print('Target GPU Device:', torch.cuda.get_device_name(0))
else:
    print('WARNING: CUDA is not available. Execution will fall back to CPU.')
