import torch
import intel_extension_for_pytorch as ipex
import segmentation_models_pytorch as smp
import time

warmup = 10
iteration_num = 100
batch_size = 1

rand_input = torch.randn(1,3,224,224)

model = smp.UnetPlusPlus()
model.eval()
model = ipex.optimize(model, dtype=torch.bfloat16)
# with torch.no_grad() and torch.cpu.amp.autocast():
#     model = torch.jit.trace(model, rand_input)
#     model = torch.jit.freeze(model)
model = torch.compile(model, backend="ipex")


with torch.no_grad():
    with torch.cpu.amp.autocast():
        for i in range(warmup):
            model(rand_input)

with torch.no_grad():
    with torch.cpu.amp.autocast():
        start = time.time()
        for i in range(iteration_num):
            model(rand_input)
        end = time.time()

print("predicton time for 100 iteration", end-start)
print("FPS", 100/(end-start))


