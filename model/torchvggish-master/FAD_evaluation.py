import torch
import numpy as np
from scipy.linalg import sqrtm

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()


output_path = ""
output = model.forward("bus_chatter.wav")
real = model.forward("2310131.wav")
print(output.shape)

def evaluation_model_forward(path):
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.eval()
    output = model.forward(path)
    return output


def calculate_frechet_distance(output, real):
    output = output.detach()
    real = real.detach()
    length = min(output.shape[0], real.shape[0])
    output = output[:length]
    real = real[:length]
    
    mu1 = output.mean(dim=0)
    mu2 = real.mean(dim=0)
    sigma1 = torch_cov(output, rowvar=False)
    sigma2 = torch_cov(real, rowvar=False)

    ssdiff = torch.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm((sigma1 @ sigma2).cpu().numpy())

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    covmean = torch.from_numpy(covmean).to(output.device)
    f_distance = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    return f_distance.numpy() 

def torch_cov(x, rowvar=False):
    if not rowvar:
        x = x.t()
    x = x - x.mean(dim=1, keepdim=True)
    n = x.size(1) - 1
    cov_matrix = (x @ x.t()).div(n)
    return cov_matrix


print(calculate_frechet_distance(output,real))