import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # audio_pickle_name = "./data/motorica_dance/kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001_01.audio29_30fps.pkl"
    # audio_pickle_name = "./data/my_music/owen_gJZ_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001_00.audio29_30fps.pkl"
    # audio_pickle_name = "./data/my_music/owen_gTP_sFM_cAll_d01_mKR_ch01_HustleandBustleofOrmos_001_00.audio29_30fps.pkl"
    audio_pickle_name = "./data/my_music/ParovStelar_gTP_sFM_cAll_d01_mKR_ch01_BootySwing_001_00.audio29_30fps.pkl"

    # expmap_pickle_name = "./data/motorica_dance/kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001_01.expmap_30fps.pkl"

    unpacked_audio = pd.read_pickle(audio_pickle_name)
    beats = unpacked_audio['Beat_0'].to_numpy()
    beats_activation = unpacked_audio['Beatactivation_0'].to_numpy()

    # Chroma_0
    # MFCC_0
    # Spectralflux_0

    # print(unpacked_audio)
    print(beats.shape)

    beats_id = np.argwhere(beats == 1)
    print(beats_id.shape)

    plt.plot(beats)
    plt.plot(beats_activation)
    plt.show()
