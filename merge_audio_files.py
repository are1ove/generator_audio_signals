import os
import random
from tqdm import tqdm
from datetime import timedelta

import scipy
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.generators import WhiteNoise


def choose_speakers(dataset_dir, num_of_speakers, min_duration, max_duration):
    """
    A function for randomly selecting a certain number of speakers from a given dataset.
    :param dataset_dir: directory with the speaker dataset
    :param num_of_speakers: number of speakers
    :param min_duration: minimum utterance length
    :param max_duration: maximum utterance length
    :return: a dictionary with speakers and file names contained in their directories
    """
    path, dirs, files = next(os.walk(dataset_dir))  # find directories with speakers in dataset
    speakers = []
    for dir in dirs:
        if dir[0].isupper():
            speakers.append(dir)  # create list only with speakers
    dirs_count = len(speakers)
    if num_of_speakers > dirs_count:  # check if user want more speakers than it has
        num_of_speakers = dirs_count

    list_of_speakers = random.sample(speakers, num_of_speakers)  # randomly select the desired number of speakers
    speakers_with_tracks = dict.fromkeys(list_of_speakers)
    for speaker in list_of_speakers:
        tracks_dir = f"{dataset_dir}/{speaker}/"
        path, dirs, files = next(os.walk(tracks_dir))  # find files in each speaker directory
        files = [f for f in files if f.split('.')[0]]
        files = sorted(files, key=lambda i: int(i.split('.')[0]))  # sort files to create right audio file
        speakers_with_tracks[speaker] = [track for track in files if
                                         min_duration <= librosa.get_duration(
                                             filename=f"{tracks_dir}/{track}") <= max_duration]  # create dict with speakers and theirs files
    return speakers_with_tracks


def merge_audio(tracks, dataset_dir, average_pause, output_duration, sample_rate):
    """
    A function that connects the sound files of each speaker together.
    It also adds pauses of a given length between speakers and connects all speakers into one audio file.
    In addition, it generates a markup file with the names of speakers and pauses with time
    :param tracks: dictionary with keys in the form of speaker names and values in the form of audio files of each speaker
    :param dataset_dir: directory with the speaker dataset
    :param average_pause: average length of pause
    :param output_duration: length of the output audio
    :param sample_rate: sampling rate
    :return: audio file with the combined statements of the speakers and pauses
    """
    res = None
    first = True
    speech_with_silence = None
    start_time = 0
    for speaker in tracks.keys():
        print(speaker)
        tracks_dir = f'{dataset_dir}/{speaker}/'
        for i in tqdm(range(0, len(tracks[speaker]) - 1)):
            if i == 0:  # if it's first file
                track1 = f'{tracks_dir}/{tracks[speaker][i]}'
                first_part = AudioSegment.from_file(track1, format="wav")  # load audio file
                first_part = first_part.set_frame_rate(sample_rate)  # change to desired sample rate
            else:
                first_part = res
            track2 = f'{tracks_dir}/{tracks[speaker][i + 1]}'
            second_part = AudioSegment.from_file(track2, format="wav")  # load audio file
            second_part = second_part.set_frame_rate(sample_rate)  # change to desired sample rate
            # TODO add mixer to audio
            res = first_part.append(second_part, crossfade=20)  # merge audio files of each speaker with crossfade
        name_of_file = f'{tracks_dir.split("/")[-2]}.wav'
        res.export(name_of_file, format="wav")  # save audio file of each speaker
        silence_segment = AudioSegment.silent(duration=average_pause)
        silence = timedelta(milliseconds=average_pause)  # create silence time part
        speech = AudioSegment.from_file(name_of_file, format="wav")  # load audio file of each speaker

        if first:  # if it's first speaker
            start_time = silence + timedelta(seconds=speech.duration_seconds)  # silence + speaker time
            speech_with_silence = silence_segment + speech + silence_segment
            with open('markup.txt', 'w') as f:  # create markup file with time
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

    speech_with_silence.export('output_test.wav', format="wav")  # save audio file with merged speakers

    return speech_with_silence


def add_noise(signal, signal_and_noise, kind_of_noise):
    """
    A function that adds noise to the input audio file with a specified signal-to-noise ratio.
    Also saves the resulting noisy signal
    :param signal: input audio file with speakers
    :param signal_and_noise: the ratio of the signal level to the noise level
    :param kind_of_noise: type of noise (white or brown)
    :return: audio file with added noise
    """
    if kind_of_noise == 'white':
        # noise = np.random.normal(0.0, 1.0, 1000)
        # signal = signal + noise #/ signal_and_noise
        noise = WhiteNoise().to_audio_segment(duration=len(signal))  # create white noise
        combined = signal.overlay(noise)  # add noise to audio
        # TODO change to signal_and_noise ratio
    elif kind_of_noise == 'brown':
        # TODO add brown noise
        noise = ''
        combined = signal.overlay(noise)
    combined.export('output_with_noise.wav', format="wav")  # save noised signal
    return combined


def add_background(speech_with_silence, background, background_level):
    """
    A function that overlays background music on the generated audio file of speakers with a specified volume level.
    Also saves the resulting signal with background
    :param speech_with_silence: the input audio file of speakers with pauses
    :param background: path to the file with the background
    :param background_level: background volume level
    :return: audio file with added background
    """
    background = AudioSegment.from_file(background, format="wav")  # take background from path

    louder_background = background + background_level

    overlay = speech_with_silence.overlay(louder_background, loop=True)  # add background to audio

    speech_with_background = overlay.export("output_with_background.wav", format="wav")  # save audio with background
    return speech_with_background


def main(dataset_dir, num_of_speakers, min_duration, max_duration, average_pause, output_duration, sample_rate,
         signal_and_noise, kind_of_noise, background, background_level):
    """
    The main function in which all other functions are called to create test audio files.
    All arguments specified by the user are passed to it.

    :param dataset_dir: directory with the speaker dataset
    :param num_of_speakers: number of speakers
    :param min_duration: minimum utterance length
    :param max_duration: maximum utterance length
    :param average_pause: average length of pauses
    :param output_duration: length of the output audio
    :param sample_rate: sampling rate
    :param signal_and_noise: ratio of the signal level to the noise level
    :param kind_of_noise: type of noise
    :param background: path to the file with the background
    :param background_level: background volume level
    :return: completion info
    """
    speakers_with_tracks = choose_speakers(dataset_dir, num_of_speakers, min_duration, max_duration)
    speech_with_silence = merge_audio(speakers_with_tracks, dataset_dir, average_pause, output_duration, sample_rate)
    speech_with_noise = add_noise(speech_with_silence, signal_and_noise, kind_of_noise)
    speech_with_background = add_background(speech_with_silence, background, background_level)
    return "Test audio files were successfully generated"


if __name__ == '__main__':
    print(
        "This script generates audio files with different speakers. "
        "You can apply noise by setting the noise type, as well as the background by specifying the path to the file "
        "with the background."
    )
    dataset_dir = str(input("Specify the directory with the speaker dataset: "))
    num_of_speakers = int(input("The number of speakers that will be present in the output audio file: "))
    min_duration = float(input("Minimum utterance length: "))
    max_duration = float(input("Maximum utterance length: "))
    average_pause = int(input("Average length of pauses between speakers in milliseconds: "))
    output_duration = int(input("The length of the output audio file in seconds: "))
    signal_and_noise = int(input("The ratio of the signal level to the noise level: "))
    kind_of_noise = str(input("Type of noise (white or brown): "))
    background = str(input("Path to the file with the background: "))
    background_level = int(input("Background volume level: "))
    sample_rate = int(input("Sampling rate of input files: "))
    print(main(dataset_dir, num_of_speakers, min_duration, max_duration, average_pause, output_duration, sample_rate,
               signal_and_noise, kind_of_noise, background, background_level))
