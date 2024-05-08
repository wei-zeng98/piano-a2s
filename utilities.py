import os
import torch
import librosa
import numpy as np
import pandas as pd
import pickle
import json
import logging
import csv
from mido import MidiFile
from hyperpyyaml import load_hyperpyyaml
from data_processing.humdrum import LabelsMultiple

labels = LabelsMultiple(extended=True)
durations = ["1","1.","2","2.","4","4.","8","8.","16","16.",
             "32","32.","64","64.","3","6","12","24","48",
             "96","128","20","40","176","112"]
duration_tokens = [labels.labels_map[d] for d in durations]
split_token = labels.labels_map["\n"]
eos_token = labels.labels_map["<eos>"]
quater_durations = [4, 6, 2, 3, 1, 3/2, 1/2, 3/4, 1/4, 3/8, 1/8, 
                    3/16, 1/16, 3/32, 4/3, 2/3, 1/3, 1/6, 1/12, 
                    1/24, 1/32, 1/5, 1/10, 1/44, 1/28]
token2quater = dict(zip(duration_tokens, quater_durations))

def load(path):
    # .npy
    if path.endswith('.npy'):
        data = np.load(path)
    
    # .json
    elif path.endswith('.json'):
        with open(path) as fin:
            data = json.load(fin)
    
    # .yaml
    elif path.endswith('.yaml'):
        with open(path) as fin:
            data = load_hyperpyyaml(fin)
            # data = SimpleNamespace(**hparams)

    # .csv
    elif path.endswith('.csv'):
        data = pd.read_csv(path, index_col=0)

    # .pkl
    elif path.endswith('.pkl'):
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
    
    # .txt
    elif path.endswith('.txt'):
        with open(path, 'r') as fin:
            data = fin.readlines()
        data = [line.strip() for line in data]

    return data

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

def create_logging(log_dir, filemode):
    mkdirs(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging

def read_metadata(csv_path):
    """Read metadata of MAESTRO dataset from csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict, dict, e.g. {
        'canonical_composer': ['Alban Berg', ...], 
        'canonical_title': ['Sonata Op. 1', ...], 
        'split': ['train', ...], 
        'year': ['2018', ...]
        'midi_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi', ...], 
        'audio_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav', ...],
        'duration': [698.66116031, ...]}
    """

    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter=',')
        lines = list(reader)

    meta_dict = {'canonical_composer': [], 'canonical_title': [], 'split': [], 
        'year': [], 'midi_filename': [], 'audio_filename': [], 'duration': []}

    for n in range(1, len(lines)):
        meta_dict['canonical_composer'].append(lines[n][0])
        meta_dict['canonical_title'].append(lines[n][1])
        meta_dict['split'].append(lines[n][2])
        meta_dict['year'].append(lines[n][3])
        meta_dict['midi_filename'].append(lines[n][4])
        meta_dict['audio_filename'].append(lines[n][5])
        meta_dict['duration'].append(float(lines[n][6]))

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])
    
    return meta_dict

def read_midi(midi_path):
    """Parse MIDI file.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            'program_change channel=0 program=0 time=0', 
            'control_change channel=0 control=64 value=127 time=0', 
            'control_change channel=0 control=64 value=63 time=236', 
            ...],
        'midi_event_time': [0., 0, 0.98307292, ...]}
    """

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 2
    """The first track contains tempo, time signature. The second track 
    contains piano events."""

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]

def traverse_folder(folder):
    paths = []
    names = []
    
    for root, dirs, files in os.walk(folder):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)
            
    return names, paths

def save(data, path):
    # .npy
    if path.endswith('.npy'):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        elif isinstance(data, list):
            data = np.array(data)
        assert isinstance(data, np.ndarray)
        np.save(path, data)

    # .json
    elif path.endswith('.json'):
        with open(path, 'w') as fout:
            json.dump(data, fout, indent=2)
    
    # .pkl
    elif path.endswith('.pkl'):
        with open(path, 'wb') as fout:
            pickle.dump(data, fout)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

def get_VQT(audio_or_path, hparams):
    if isinstance(audio_or_path, str):
        y, fs = librosa.load(audio_or_path, sr=hparams["sample_rate"])
    else:
        y = audio_or_path
        fs = hparams["sample_rate"]
    vqt = librosa.vqt(y, 
                      sr=fs, 
                      hop_length=hparams["hop_length"], 
                      fmin=librosa.note_to_hz('A0'), 
                      n_bins=hparams["bins_per_octave"] * hparams["n_octaves"], 
                      bins_per_octave=hparams["bins_per_octave"], 
                      gamma=hparams["gamma"])
    log_vqt = ((1./80.) * librosa.amplitude_to_db(np.abs(np.array(vqt)), ref=np.max)) + 1.
    return log_vqt.T # (frames_num, bins_num)

def get_sequence_duration(sequence):
    """Get the quantized duration of a kern sequence.

    Args:
      sequence: list of tokens

    Returns:
      duration: quarter note duration
    """
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
    quarter_duration = 0
    new_line = True
    for token in sequence:
        if token == eos_token:
            break
        if token == split_token:
            new_line = True
        if token in duration_tokens:
            if new_line:
                quarter_duration += token2quater[token]
                new_line = False
    return quarter_duration

class MIDIProcess(object):
    def __init__(self, midi_path, split='train'):
        self.midi = MidiFile(midi_path)
        assert split in ['train', 'valid', 'test']
        self.split = split

    def cut_last_pedal(self):
        for track in self.midi.tracks:
            if track[-2].type == 'control_change' and track[-2].channel == 0 \
                and track[-2].control == 64 and track[-2].value == 0:
                track[-2].time = 0

    def cut_initial_blank(self):
        total_time_before_first_note = 0
        found_first_note = False

        for track in self.midi.tracks:
            time_accumulated = 0
            for msg in track:
                if not found_first_note:
                    time_accumulated += msg.time
                    if (msg.type == 'note_on' and msg.velocity > 0) or (msg.type == 'control_change' and msg.value > 0):
                        found_first_note = True
                        total_time_before_first_note = time_accumulated - msg.time
                        msg.time = 0
                else:
                    msg.time -= total_time_before_first_note
                    break

    def ramdom_scaling(self, range=(0.85, 1.15)):
        original_length = self.midi.length
        lower_bound = max(range[0], 4. / original_length)
        upper_bound = min(range[1], 12. / original_length)
        if lower_bound > upper_bound:
            return None, original_length
        if self.split == 'test' or self.split == 'valid':
            if lower_bound > 1:
                scaling = lower_bound
            elif upper_bound < 1:
                scaling = upper_bound
            else:
                scaling = 1
        elif self.split == 'train':
            scaling = np.random.uniform(lower_bound, upper_bound)
        for track in self.midi.tracks:
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off' or msg.type == 'control_change' or msg.type == 'program_change':
                    msg.time = int(msg.time * scaling)
        return scaling, original_length
    
    def save(self, path):
        try:
            self.midi.save(path)
        except:
            print('Error in saving midi file {}'.format(path))
    
    def process(self, path):
        self.cut_last_pedal()
        self.cut_initial_blank()
        # Save to get correct length
        self.midi.save('temp/temp.mid')
        self.midi = MidiFile('temp/temp.mid')
        scaling, original_length = self.ramdom_scaling()
        if scaling is not None:
            self.save(path)
        return scaling, original_length