import librosa, librosa.display
import matplotlib.pyplot as plt
import io
import math
import statistics


hop_length = 512
n_fft = 2048
num_mfcc = 14


def add(a, b):
    return a+b

def get_mfcc_image(file_path):
    signal, sample_rate = librosa.load(file_path, sr=22050, offset=30, duration=3)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=num_mfcc)

    plt.figure(figsize=(5,3))
    librosa.display.specshow(mfcc, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()

    bio = io.BytesIO()
    plt.savefig(bio, format="png")
    b = bio.getvalue()

    return b

def get_pyin(file_path):
    signal, sample_rate = librosa.load(file_path, sr=22050, offset=30, duration=3)
    f0, voiced_flag, voiced_probs = librosa.pyin(signal,
                                                 fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7'),
                                                 sr=sample_rate)

    f0.sort()

    median_val = statistics.median(f0)
    max_val = f0[0]

    return (max_val + median_val)/2, f0.tolist()


def generate_mfcc(file_path):
    signal, sample_rate = librosa.load(file_path, sr=22050, offset=30, duration=3)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=14)

    plt.figure(figsize=(5,3))
    librosa.display.specshow(mfcc, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()

    bio = io.BytesIO()
    plt.savefig(bio, format="png")
    b = bio.getvalue()

    return b, mfcc


def get_all_mfcc(file_path):

    all_mfcc = []

    signal, sample_rate = librosa.load(file_path, sr=22050)
    duration = librosa.get_duration(signal)

    samples_per_track = sample_rate * duration

    num_segments = int(duration / 3)

    samples_per_segment = int(samples_per_track / num_segments)

    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for d in range(num_segments):

        start = samples_per_segment * d
        finish = start + samples_per_segment

        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        if len(mfcc) == num_mfcc_vectors_per_segment:
            all_mfcc.append(mfcc)


    return all_mfcc



def get_tempo(file_path):
    signal, sample_rate = librosa.load(file_path, sr=22050, offset=30, duration=3)
    onset_env = librosa.onset.onset_strength(y=signal, sr=sample_rate)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)

    return tempo[0].item()

