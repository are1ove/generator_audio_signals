import os
import random
from tqdm import tqdm
from datetime import timedelta

import scipy
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.generators import WhiteNoise


def choose_speakers(dataset_dir, num_of_speakers, min_duration, max_duration):
    path, dirs, files = next(os.walk(dataset_dir))
    speakers = []
    for dir in dirs:
        if dir[0].isupper():
            speakers.append(dir)
    dirs_count = len(speakers)
    if num_of_speakers > dirs_count:
        num_of_speakers = dirs_count

    list_of_speakers = random.sample(speakers, num_of_speakers)
    speakers_with_tracks = dict.fromkeys(list_of_speakers)
    for speaker in list_of_speakers:
        tracks_dir = f"{dataset_dir}/{speaker}/"
        path, dirs, files = next(os.walk(tracks_dir))
        files = [f for f in files if f.split('.')[0]]
        files = sorted(files, key=lambda i: int(i.split('.')[0]))
        speakers_with_tracks[speaker] = [track for track in files if
                                         min_duration <= librosa.get_duration(
                                             filename=f"{tracks_dir}/{track}") <= max_duration]
    return speakers_with_tracks


def merge_audio(tracks, dataset_dir, average_pause, output_duration, sample_rate):
    res = None
    first = True
    speech_with_silence = None
    start_time = 0
    for speaker in tracks.keys():
        print(speaker)
        tracks_dir = f'{dataset_dir}/{speaker}/'
        for i in tqdm(range(0, len(tracks[speaker]) - 1)):
            if i == 0:
                track1 = f'{tracks_dir}/{tracks[speaker][i]}'
                x, s_r = librosa.load(track1, duration=5)
                first_part = librosa.resample(x, s_r, sample_rate)
            else:
                first_part = res
            track2 = f'{tracks_dir}/{tracks[speaker][i + 1]}'
            y, s_r = librosa.load(track2, duration=5)
            second_part = librosa.resample(y, s_r, sample_rate)
            # TODO add mixer to audio
            res = np.append(first_part, second_part)
        name_of_file = f'{tracks_dir.split("/")[-2]}.wav'
        sf.write(name_of_file, res, sample_rate)
        silence_segment = AudioSegment.silent(duration=average_pause)
        silence = timedelta(milliseconds=average_pause)
        speech = AudioSegment.from_wav(name_of_file)

        if first:
            start_time = silence + timedelta(seconds=speech.duration_seconds)
            speech_with_silence = silence_segment + speech + silence_segment
            with open('markup.txt', 'w') as f:
                f.write(f"{timedelta()} - silence\n")
                f.write(f"{silence} - {name_of_file}\n")
                f.write(f"{start_time} - silence\n")
            first = False
        else:
            speech_with_silence = speech_with_silence + speech + silence_segment
            with open('markup.txt', 'a') as f:
                f.write(f"{start_time + silence} - {name_of_file}\n")
                f.write(f"{start_time + silence + timedelta(seconds=speech.duration_seconds)} - silence\n")
            start_time = start_time + timedelta(seconds=speech.duration_seconds) + silence
        # TODO compare output audio duration and desired time

    speech_with_silence.export('output_test.wav', format="wav")

    return speech_with_silence


def add_noise(signal, signal_and_noise, kind_of_noise):
    # TODO add_noise
    if kind_of_noise == 'white':
        # noise = np.random.normal(0.0, 1.0, 1000)
        # signal = signal + noise #/ signal_and_noise
        noise = WhiteNoise().to_audio_segment(duration=len(signal))
        combined = signal.overlay(noise)
    elif kind_of_noise == 'brown':
        exponent = 2
        fmin = 0
        size = [5]

        samples = size[-1]

        f = np.fft.rfftfreq(samples)

        s_scale = f
        fmin = max(fmin, 1. / samples)
        ix = np.sum(s_scale < fmin)
        if ix and ix < len(s_scale):
            s_scale[:ix] = s_scale[ix]
        s_scale = s_scale ** (-exponent / 2.)

        w = s_scale[1:].copy()
        w[-1] *= (1 + (samples % 2)) / 2.
        sigma = 2 * np.sqrt(np.sum(w ** 2)) / samples

        size[-1] = len(f)

        dims_to_add = len(size) - 1
        s_scale = s_scale[(np.newaxis,) * dims_to_add + (Ellipsis,)]

        sr = np.random.normal(scale=s_scale, size=size)
        si = np.random.normal(scale=s_scale, size=size)

        if not (samples % 2): si[..., -1] = 0

        si[..., 0] = 0

        s = sr + 1J * si
        noise = np.fft.irfft(s, n=samples, axis=-1) / sigma

        combined = signal + noise  # / signal_and_noise
    combined.export('output_with_noise.wav', format="wav")
    return combined


def add_background(background, background_level):
    # TODO add_background
    pass


def main(dataset_dir, num_of_speakers, min_duration, max_duration, average_pause, output_duration, sample_rate,
         signal_and_noise, kind_of_noise):
    speakers_with_tracks = choose_speakers(dataset_dir, num_of_speakers, min_duration, max_duration)
    speech_with_silence = merge_audio(speakers_with_tracks, dataset_dir, average_pause, output_duration, sample_rate)
    speech_with_noise = add_noise(speech_with_silence, signal_and_noise, kind_of_noise)


if __name__ == '__main__':
    dataset_dir = str(input("dataset_dir: "))
    num_of_speakers = int(input("num_of_speakers: "))
    min_duration = float(input("min_duration: "))
    max_duration = float(input("max_duration: "))
    average_pause = int(input("average_pause in ms: "))
    output_duration = int(input("output_duration in s: "))
    signal_and_noise = int(input("signal_and_noise: "))
    kind_of_noise = str(input("kind_of_noise: "))
    # background = str(input("background: "))
    # background_level = int(input("background_level: "))
    sample_rate = int(input("sample_rate: "))
    main(dataset_dir, num_of_speakers, min_duration, max_duration, average_pause, output_duration, sample_rate,
         signal_and_noise, kind_of_noise)
