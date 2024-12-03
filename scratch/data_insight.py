import numpy as np

data_path = "data/msmd_aug_v1-1_no-audio/BachJS__BWV117a__BWV-117a/scores/BachJS__BWV117a__BWV-117a_ly/coords/notes_01.npy"
file = np.load(data_path)
print(file)