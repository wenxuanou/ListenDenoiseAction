import os
import sys

import pickle as pkl
import madmom
from utils.cepstrogram import MFCCProcessor

# TODO: extract beat activation using madmom.features.downbeats

# TODO: extract MFCC using madmom.audio.cepstrogram

# TODO: extract chroma using madmom.audio.chroma

if __name__ == "__main__":
    audio_name = "data/motorica_dance/kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001.wav"

    # read in file
    audio = madmom.audio.signal.Signal('data/sample.wav')
    print("loaded audio file, shape: ", audio.shape)
    print("signal sample rate: ", audio.sample_rate)

    # get spectrogram
    spec = madmom.audio.spectrogram.SpectrogramProcessor().process(audio)
    print("get spectrogram, shape: ", spec.shape)
    # get spectral flux
    sf = madmom.features.onsets.spectral_flux(spec)
    print("get spectral flux, shape: ", sf.shape)

    # get chroma, use CLP chroma for now, TODO: CLP Chroma is missing the official code
    chroma = madmom.audio.CLPChromaProcessor(fps=30).process(audio)
    print("get chroma, shape: ", chroma.shape)

    # get MFCC
    MFCC = MFCCProcessor().process(spec)
    print("get MFCC, shape: ", MFCC.shape)

    # TODO: sample rate should be 30 fps

    # TODO: output pickle should be of size: (N, 29), N as number of frame
    # with 20 MFCC + 6 chroma + 1 beat + 1 beatactivation + 1 spectralflux
    # items with posefix: MFCC_0, beat_0, .....