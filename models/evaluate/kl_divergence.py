import librosa
import numpy as np
from scipy.stats import entropy
import os
def load_audio_features(file_path):
    audio, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)
    mfccs_flattened = mfccs.flatten()
    return mfccs_flattened

def compute_kl_divergence(p, q):
     nonzero_indices = (p != 0) & (q != 0)
     return np.sum(p[nonzero_indices] * np.log(p[nonzero_indices] / q[nonzero_indices]))

def create_histogram(features, bins=30, smoothing_constant=1e-10):
    hist, bin_edges = np.histogram(features, bins=bins, density=True)
    hist = hist / np.sum(hist)
    return hist
kl = []
import os

def list_files(directory):
    """ List all files in a directory (not including subdirectories) """
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

# Paths to your directories
dir1_path = 'test_data/im2wav_split'
dir2_path = 'test_data/im4wav_split'
ground_path = 'test_data/ground_truth_split'

dir1_files = list_files(dir1_path)
dir2_files = list_files(dir2_path)
ground_files = list_files(ground_path)
for file1, file2 in zip(ground_files, dir2_files):
  file1_path = os.path.join(ground_path, file1)
  file2_path = os.path.join(dir2_path, file2)
  original_features = load_audio_features(file1_path)
  generated_features = load_audio_features(file2_path)

  original_hist = create_histogram(original_features)
  generated_hist = create_histogram(generated_features)
  min_bins = min(len(original_hist), len(generated_hist))
  original_hist = original_hist[:min_bins]
  generated_hist = generated_hist[:min_bins]
  kl_divergence = compute_kl_divergence(original_hist, generated_hist)
  kl.append(kl_divergence)
print(kl)
print(sum(kl)/len(kl))