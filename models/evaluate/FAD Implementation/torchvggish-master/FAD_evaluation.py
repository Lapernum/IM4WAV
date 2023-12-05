import torch
import numpy as np
from scipy.linalg import sqrtm
import os
from pydub import AudioSegment 

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()
list1 = os.listdir("ground_truth")
list2 = os.listdir("l1")


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



list1111 = []
list2222 = []
count = 0
try:
    for name in list1:
        # if count in [2, 3, 8]:
        #     count += 1
        #     continue
        if (name == "0_car-engine-starting-9_0.wav"):
            continue
        path1 = "ground_truth/" + name
        path2 = "combined_audio/" + name
        stereo_audio1 = AudioSegment.from_file( 
            path1,
            format = "wav") 
        stereo_audio2 = AudioSegment.from_file( 
            path2, 
            format="wav") 
        mono_audios1 = stereo_audio1.split_to_mono()
        mono_audios2 = stereo_audio2.split_to_mono()
        mono_audios11 = None
        mono_audios12 = None
        mono_audios21 = None
        mono_audios22 = None
        if len(mono_audios1) == 2:
            mono_audios11 = mono_audios1[0].export( 
            path1[:-4] + "_left.wav", 
            format="wav") 
            mono_audios12 = mono_audios1[1].export( 
            path1[:-4] + "_right.wav", 
            format="wav") 
        else:
            continue
            mono_audios11 = mono_audios1[0].export( 
            path1[:-4] + "_left.wav", 
            format="wav") 

            mono_audios12 = mono_audios1[0].export( 
            path1[:-4] + "_left.wav", 
            format="wav") 
        if len(mono_audios2) == 2:
            mono_audios21 = mono_audios2[0].export( 
            path2[:-4] + "_left.wav", 
            format="wav") 
            mono_audios22 = mono_audios2[1].export( 
            path2[:-4] + "_right.wav", 
            format="wav") 
        else:
            mono_audios21 = mono_audios2[0].export( 
            path2[:-4] + "_left.wav", 
            format="wav") 
            mono_audios22 = mono_audios2[0].export( 
            path2[:-4] + "_right.wav", 
            format="wav") 
        output_left = model.forward(path1[:-4] + "_left.wav")
        output_right = model.forward(path1[:-4] + "_right.wav")
        fid1 = calculate_frechet_distance(output_left, output_right)
        if fid1 < -600:
            print(fid1)
            print(path1)
            print(path2)
        else:
            output11 = model.forward(path1[:-4] + "_left.wav")
            output12 = model.forward(path1[:-4] + "_right.wav")
            output21 = model.forward(path2[:-4] + "_left.wav")
            output22 = model.forward(path2[:-4] + "_right.wav")
            list1111.append(calculate_frechet_distance(output11, output21))
            list2222.append(calculate_frechet_distance(output12, output22))

except:
    pass
    
    # Calling the split_to_mono method 
    # on the stereo audio file 
print(list1111)
print(list2222)
print(len(list2222))
sum1 = sum(list1111) / (len(list1111))
sum2 = sum(list2222) / (len(list2222))
print(sum(list1111) / (len(list1111)))
print(sum(list2222) / (len(list2222)))
print((sum1 + sum2)/2)