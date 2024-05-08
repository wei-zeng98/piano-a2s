import os
import torch
from tqdm import tqdm
import pretty_midi as pm
from torch.utils.data import Dataset, DataLoader
from utilities import save, load
from data_processing.humdrum import LabelsMultiple
import numpy as np

class SyntheticDataset(Dataset):
    def __init__(self, hparams, split, device, version=[0]):
        """Combined Dataset from MuseSyn and Humdrum database.

        Args:
            hparams: (types.SimpleNamespace) hyperparameters.
            split: (str) split of the dataset.
            """
        super().__init__()
        self.hparams = hparams
        self.split = split
        self.device = device
        self.version = version
        self.song_list, self.lengths = self.get_song_list()
        self.time_sig_list = load('data_processing/metadata/time_signature_list.json')
        self.time_sig_dict = {time_sig: i for i, time_sig in enumerate(self.time_sig_list)}
        self.labels = LabelsMultiple(extended=True)

    def get_song_list(self):
        song_list = {}
        lengths = {}
        for v in self.version:
            feature_folder = os.path.join(self.hparams["feature_folder"], f'{self.split}', f'{v}')
            song_list[v] = [song[:-4] for song in os.listdir(os.path.join(feature_folder, 'spectrogram'))]
            song_list[v].sort()
            lengths[v] = len(song_list[v])
        return song_list, lengths

    def key_to_int(self, key_signatures):
        key_signatures = torch.tensor(key_signatures).to(self.device) + 6
        return key_signatures
    
    def time_sig_to_int(self, time_signatures):
        time_signatures = torch.tensor([self.time_sig_dict[time_sig] for time_sig in time_signatures]).to(self.device)
        return time_signatures

    def pad_spectrogram(self, spectrogram):
        """Pad spectrogram to the same length.
            
            Args:
                spectrogram: (np.ndarray) spectrogram. (T, n_mels)
            """
        
        max_len = self.hparams["max_frame_num"]
        spectrogram = torch.from_numpy(spectrogram).float()
        padded_spectrogram = torch.zeros((max_len, spectrogram.shape[-1]))
        num_frames = min(spectrogram.shape[0], max_len)
        padded_spectrogram[:num_frames] = spectrogram
        return padded_spectrogram.unsqueeze(0) # (1, T, n_mels)

    def pad_score(self, score, max_length):
        padded_score = torch.zeros((len(score), max_length), dtype=torch.long).to(self.device)
        lengths = torch.tensor([min(len(measure), max_length) for measure in score]).to(self.device) # (num_bars,)
        for i, measure in enumerate(score):
            padded_score[i] = self.pad_single_measure(measure, max_length)
        return padded_score, lengths  # (num_bars, max_length), (num_bars,)

    def pad_single_measure(self, measure, max_length):
        padded_measure = torch.ones((max_length,), dtype=torch.long) * self.labels.labels_map['<pad>']
        if len(measure) > max_length:
            measure = measure[:max_length]
        padded_measure[:len(measure)] = torch.tensor(measure)
        if len(measure) < max_length:
            padded_measure[len(measure)] = self.labels.labels_map['<eos>']
        return padded_measure.to(self.device) # (max_length,)

class TrainDataset(SyntheticDataset):
    def __init__(self, hparams, split, device, version=[0]):
        """
        Args:
            hparams: (types.SimpleNamespace) hyperparameters.
            split: (str) split of the dataset.
            """
        super().__init__(hparams, split, device, version)

    def __len__(self):
        return max(self.lengths.values())

    def __getitem__(self, idx):
        # Get random version
        v = self.version[np.random.randint(len(self.version))]
        feature_folder = os.path.join(self.hparams["feature_folder"], f'{self.split}', f'{v}')
        song_list = self.song_list[v]
        length = self.lengths[v]
        idx = idx % length
        spectrogram_name = song_list[idx]
        target_name = spectrogram_name.split('~')[0]

        # Get spectrogram
        spectrogram_file = os.path.join(feature_folder, 'spectrogram', f'{spectrogram_name}.npy')
        spectrogram = self.pad_spectrogram(load(spectrogram_file))  # (1, T, n_mels)

        # Get score
        score_file = os.path.join(feature_folder, 'target', f'{target_name}.pkl')
        score = load(score_file)

        key_signatures = self.key_to_int([item[0] for item in score])
        time_signatures = self.time_sig_to_int([item[1] for item in score])
        upper_staff = [item[3] for item in score]
        lower_staff = [item[2] for item in score]
        upper_staff = self.pad_score(upper_staff, self.hparams["max_length"][0])
        lower_staff = self.pad_score(lower_staff, self.hparams["max_length"][1])

        return spectrogram.to(self.device), \
               time_signatures, \
               key_signatures, \
               upper_staff[0], \
               upper_staff[1], \
               lower_staff[0], \
               lower_staff[1], \
               spectrogram_name, \
               v

class TestDataset(SyntheticDataset):
    def __init__(self, hparams, split, device, version=[0]):
        """
        Args:
            hparams: (types.SimpleNamespace) hyperparameters.
            split: (str) split of the dataset.
            """
        super().__init__(hparams, split, device, version)
        self.song_list = []
        for v in version:
            feature_folder = os.path.join(self.hparams["feature_folder"], f'{self.split}', f'{v}')
            for song in os.listdir(os.path.join(feature_folder, 'spectrogram')):
                self.song_list.append([song[:-4], v])
        self.length = len(self.song_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get item
        spectrogram_name, v = self.song_list[idx]
        target_name = spectrogram_name.split('~')[0]
        feature_folder = os.path.join(self.hparams["feature_folder"], f'{self.split}', f'{v}')

        # Get spectrogram
        spectrogram_file = os.path.join(feature_folder, 'spectrogram', f'{spectrogram_name}.npy')
        spectrogram = self.pad_spectrogram(load(spectrogram_file))  # (1, T, n_mels)

        # Get score
        score_file = os.path.join(feature_folder, 'target', f'{target_name}.pkl')
        score = load(score_file)

        key_signatures = self.key_to_int([item[0] for item in score])
        time_signatures = self.time_sig_to_int([item[1] for item in score])
        upper_staff = [item[3] for item in score]
        lower_staff = [item[2] for item in score]
        upper_staff = self.pad_score(upper_staff, self.hparams["max_length"][0])
        lower_staff = self.pad_score(lower_staff, self.hparams["max_length"][1])

        return spectrogram.to(self.device), \
               time_signatures, \
               key_signatures, \
               upper_staff[0], \
               upper_staff[1], \
               lower_staff[0], \
               lower_staff[1], \
               spectrogram_name, \
               v

if __name__ == '__main__':
    hparams = load('hparams/pretrain.yaml')
    dataset = TrainDataset(hparams, 'train', 'cuda:#', range(10))
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(len(train_loader))
    for i, batch in tqdm(enumerate(train_loader)):
        spectrogram, time_sig_target, key_target, upper_target, upper_lengths, \
            lower_target, lower_lengths, song_name, version = batch
        print(time_sig_target.shape)
        print(time_sig_target)
        break