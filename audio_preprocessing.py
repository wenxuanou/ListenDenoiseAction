import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import madmom
from utils.cepstrogram import MFCC  # the processor is buggy, use function directly

DANCE_STYLE = {
    "Hiphop": "gLH",
    "Krumping": "gKR",
    "Popping": "gPO",
    "Locking": "gLO",
    "Jazz": "gJZ",
    "Charleston": "gCH",
    "Tapping": "gTP",
    "Casual": "gCA",
}

def generate_output_name(save_dir: str, author: str, audio_name: str, dance_style: str):
    # kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001_00.audio29_30fps.pkl

    style_code = DANCE_STYLE[dance_style]
    filename = author + "_" \
               + style_code + "_" \
               + "sFM_cAll_d01_mKR_ch01" + "_" \
               + audio_name + "_" \
               + "001_00" + ".audio29_30fps.pkl"   # 29 features audio, 30 fps

    return os.path.join(save_dir, filename)


if __name__ == "__main__":
    audio_name = "data/motorica_dance/kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001.wav"

    # read in file
    audio = madmom.audio.signal.Signal(audio_name)
    print("loaded audio file, shape: ", audio.shape)
    print("signal sample rate: ", audio.sample_rate)

    sample_num_30fps = int(np.rint(audio.shape[0] / audio.sample_rate * 30.0))   # # of samples in 30 fps, round to int
    print("# of samples in 30fps: ", sample_num_30fps)
    time_index = np.arange(sample_num_30fps) / 30.0         # in seconds
    time_index = pd.TimedeltaIndex(time_index, unit="s")

    # get RNN down beat activation
    downbeat_activation = madmom.features.RNNDownBeatProcessor(fps=30)(audio)
    # beats_pred = madmom.features.DBNDownBeatTrackingProcessor([1], fps=30)(downbeat_activation)
    print("get beat activations, shape: ", downbeat_activation.shape)

    beats_activation = madmom.features.RNNBeatProcessor(fps=30)(audio)
    beats_pred = madmom.features.BeatTrackingProcessor(fps=30)(beats_activation)
    print("get beat predictions, shape: ", beats_pred.shape)    # column 0: beat position (sec), column 1: beat number in position

    beats = np.arange(sample_num_30fps)
    for i in range(sample_num_30fps):
        beats[i] = (beats[i] / 30.0) in beats_pred

    # plt.figure(1)
    # plt.plot(downbeat_activation[:, 0])
    # plt.plot(beats)
    # plt.figure(2)
    # plt.plot(beats_activation)
    # plt.plot(beats)
    # plt.show()

    # get chroma, use CLP chroma for now, TODO: CLP Chroma is missing in the official code
    chroma = madmom.audio.CLPChromaProcessor(fps=30).process(audio)   # TODO: fps=30?  # why use only 6 not 12 chroma?
    chroma = chroma[0:sample_num_30fps, :]                            # make sure align
    print("get chroma, shape: ", chroma.shape)

    # get spectrogram
    spec = madmom.audio.spectrogram.Spectrogram(audio, fps=30)
    print("get spectrogram, shape: ", spec.shape)

    # get spectral flux
    sf = madmom.features.onsets.spectral_flux(spec)
    print("get spectral flux, shape: ", sf.shape)

    # get MFCC
    mfcc = MFCC(spec, num_bands=20)
    print("get MFCC, shape: ", mfcc.shape)


    # TODO: output pickle should be of size: (N, 29), N as number of frame
    # with 20 MFCC + 6 chroma + 1 beat + 1 beatactivation + 1 spectralflux
    # items with posefix: MFCC_0, beat_0, .....

    # building dataframe
    series_dict = {}
    series_dict["Beat_0"] = pd.Series(beats, index=time_index)
    series_dict["Beatactivation_0"] = pd.Series(downbeat_activation[:, 1], index=time_index)
    series_dict["Spectralflux_0"] = pd.Series(sf, index=time_index)

    for i in range(chroma.shape[1]):     # should be 6 in official dataset, here use all
        series_dict[f"Chroma_{i}"] = pd.Series(chroma[:, i], index=time_index)

    for i in range(mfcc.shape[1]):
        series_dict[f"MFCC_{i}"] = pd.Series(mfcc[:, i], index=time_index)

    df = pd.DataFrame.from_dict(series_dict)
    # print("get dataframe")

    output_name = generate_output_name("data/motorica_dance", "owen", "chargedcableupyour", "Jazz")
    print("output file name: ", output_name)
    df.to_pickle(output_name)