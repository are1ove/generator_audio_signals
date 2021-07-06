import os
import random
from tqdm import tqdm

import scipy
import librosa
import numpy as np
import soundfile as sf


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
        files = sorted(files, key=lambda i: int(i.split('.')[0]))
        speakers_with_tracks[speaker] = [track for track in files if
                                         min_duration <= librosa.get_duration(
                                             filename=f"{tracks_dir}/{track}") <= max_duration]
    return speakers_with_tracks


def merge_audio(tracks, dataset_dir):
    res = None
    sample_rate = None
    for speaker in tracks.keys():
        print(speaker)
        tracks_dir = f'{dataset_dir}/{speaker}/'
        for i in range(0, len(tracks[speaker]) - 1):
            if i == 0:
                track1 = f'{tracks_dir}/{tracks[speaker][i]}'
                first_part, sample_rate = librosa.load(track1, duration=5)
            else:
                first_part = res
            track2 = f'{tracks_dir}/{tracks[speaker][i + 1]}'
            second_part, sample_rate = librosa.load(track2, duration=5)
            # TODO add micsher to audio
            res = np.append(first_part, second_part)

        sf.write(f'{tracks_dir.split("/")[-2]}.wav', res, sample_rate)
        # TODO add pause after each speaker
        # TODO write speaking time in markup file
        # TODO merge speakers

    return merged_audio, markup_file


def add_noise(noise_file, signal_and_noise):
    pass


def add_background(background, background_level):
    pass


def main(dataset_dir, num_of_speakers, min_duration, max_duration):
    speakers_with_tracks = choose_speakers(dataset_dir, num_of_speakers, min_duration, max_duration)
    merge_audio(speakers_with_tracks, dataset_dir)


def change_rate(sample_rate):
    pass


if __name__ == '__main__':
    dataset_dir = str(input("dataset_dir: "))
    num_of_speakers = int(input("num_of_speakers: "))
    min_duration = float(input("min_duration: "))
    max_duration = float(input("max_duration: "))
    # average_pause = int(input("average_pause: "))
    # output_duration = int(input("output_duration: "))
    # signal_and_noise = int(input("signal_and_noise: "))
    # noise_file = str(input("noise_file: "))
    # background = str(input("background: "))
    # background_level = int(input("background_level: "))
    # sample_rate = float(input("sample_rate: "))
    main(dataset_dir, num_of_speakers, min_duration, max_duration)

    # tracks = ['0.wav', '1.wav', '2.wav', '3.wav', '4.wav', '5.wav']
    # tracks_dir = '/Users/valeriy/Documents/sMedX/generator_test_signals/16000_pcm_speeches/Benjamin_Netanyau'
    # merge_audio(list_of_speakers['Benjamin_Netanyau'], dataset_dir)
