import os
import sys

import pickle as pkl

if __name__ == "__main__":
    audio_pickle_name = "./data/motorica_dance/kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001_00.audio29_30fps.pkl"
    expmap_pickle_name = "./data/motorica_dance/kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001_00.expmap_30fps.pkl"

    with open(audio_pickle_name, 'rb') as f:
        audio_pickle = pkl.load(f)

    with open(expmap_pickle_name, 'rb') as f:
        expmap_pickle = pkl.load(f)

    print("audio pickle: ")
    print(vars(audio_pickle))

    print("expmap pickle: ")
    print(vars(expmap_pickle))

    print("done")