import os
import sys

import pickle as pkl
import madmom
from utils.cepstrogram import MFCC  # the processor is buggy, use function directly

# TODO: extract beat activation using madmom.features.downbeats

# TODO: extract MFCC using madmom.audio.cepstrogram

# TODO: extract chroma using madmom.audio.chroma

if __name__ == "__main__":
    audio_name = "data/motorica_dance/kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001.wav"

    # read in file
    audio = madmom.audio.signal.Signal(audio_name)
    print("loaded audio file, shape: ", audio.shape)
    print("signal sample rate: ", audio.sample_rate)

    # get RNN down beat activation
    beat_proc = madmom.features.RNNDownBeatProcessor()  # default sampling rate: 100 fps
    beat_activation = beat_proc(audio)     # column 0 is beat, column 1 is downbeat
    print("get down beat activate, shape: ", beat_activation.shape)

    # get chroma, use CLP chroma for now, TODO: CLP Chroma is missing the official code
    chroma = madmom.audio.CLPChromaProcessor(fps=30).process(audio)   # TODO: fps=30?
    print("get chroma, shape: ", chroma.shape)

    # get spectrogram
    spec = madmom.audio.spectrogram.SpectrogramProcessor().process(audio)
    print("get spectrogram, shape: ", spec.shape)

    # get spectral flux
    sf = madmom.features.onsets.spectral_flux(spec)
    print("get spectral flux, shape: ", sf.shape)

    # get MFCC
    mfcc = MFCC(spec)
    print("get MFCC, shape: ", mfcc.shape)

    # TODO: sample rate should be 30 fps

    # TODO: output pickle should be of size: (N, 29), N as number of frame
    # with 20 MFCC + 6 chroma + 1 beat + 1 beatactivation + 1 spectralflux
    # items with posefix: MFCC_0, beat_0, .....