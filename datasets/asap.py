import os
import pandas as pd
import torch
from tqdm import tqdm
import pretty_midi as pm
from torch.utils.data import Dataset, DataLoader
from utilities import save, load, mkdirs, get_VQT
from data_processing.humdrum import LabelsMultiple
import numpy as np
from pathlib import Path
from data_processing.humdrum import Kern, sort_chords, sort_voices, process_voices
import subprocess
import music21 as m21
import torchaudio
from data_processing.humdrum import LabelsMultiple

labels = LabelsMultiple(extended=True)
SOS = labels.labels_map['<sos>']
EOS = labels.labels_map['<eos>']

class ProcessASAP(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.asap_folder = hparams["asap_folder"]
        self.feature_folder = hparams["feature_folder"]
        self.folders = self._get_smallest_subdirectories()
        self.train_songs = set([row['name'] for i, row in \
                                pd.read_csv('data_processing/metadata/train_asap.txt').iterrows()])
        self.test_songs = set([row['name'] for i, row in \
                               pd.read_csv('data_processing/metadata/test_asap.txt').iterrows()])
        self.time_sig_list = load('data_processing/metadata/time_signature_list.json')

    def process_all(self):
        # Make folders
        for split in ['train', 'test']:
            mkdirs(f'temp/{split}')
            for folder in ['wav', 'midi', 'xml', 'kern', 'target', 'kern_upper', 'kern_lower', 'info']:
                mkdirs(f'{self.feature_folder}/{split}/{folder}')

        unmatched = []
        for folder in tqdm(self.folders):
            unmatched.extend(self.process_one(folder))
        
        with open('unmatched.txt', 'w') as f:
            for item in unmatched:
                f.write("%s\n" % item)
        
        self._prepare_spectrograms()
    
    def process_one(self, folder):
        # Get score name
        score_name = self._get_score_name_from_folder(folder)
        if score_name in self.train_songs:
            split = 'train'
        elif score_name in self.test_songs:
            split = 'test'
        else:
            return []
        # Get score
        m21_score = m21.converter.parse(os.path.join(folder, 'xml_score.musicxml'))
        n_measure_score = len(m21_score.parts[0].getElementsByClass('Measure'))
        
        # Split m21 score into chunks
        chunks = []
        for i in range(1, n_measure_score - 4):
            chunk = m21_score.measures(i, i + 4)
            chunks.append(chunk)
        # Process performances
        performances = [file[:-4] for file in os.listdir(folder) if file.endswith('.wav')]
        unmatched = []
        for performance in performances:
            # Check if number of measures match
            anno_file = os.path.join(folder, f'{performance}_annotations.txt')
            upbeat, downbeats = self._get_anno_downbeats(anno_file)
            n_measure_annotation = len(downbeats) if upbeat else len(downbeats) - 1
            if n_measure_score != n_measure_annotation:
                unmatched.append('#'.join([score_name, performance]))
                continue
            feature_folder = os.path.join(self.feature_folder, split)
            audio, sample_rate = torchaudio.load(os.path.join(folder, f'{performance}.wav'))
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            # Normalize
            audio = audio / torch.max(torch.abs(audio))

            # Cut and save
            for i, chunk in enumerate(chunks):
                if upbeat and i == 0: continue
                wav_path = os.path.join(feature_folder, 'wav', f'{score_name}#{performance}.{i}.wav')
                xml_path = os.path.join(feature_folder, 'xml', f'{score_name}#{performance}.{i}.xml')
                kern_path = os.path.join(feature_folder, 'kern', f'{score_name}#{performance}.{i}.krn')
                target_path = os.path.join(feature_folder, 'target', f'{score_name}#{performance}.{i}.pkl')
                lower_path = os.path.join(feature_folder, 'kern_lower', f'{score_name}#{performance}.{i}.krn')
                upper_path = os.path.join(feature_folder, 'kern_upper', f'{score_name}#{performance}.{i}.krn')
                # Save wav
                try:
                    chunk_audio = audio[:, int(downbeats[i+1][0] * sample_rate): int(downbeats[i+6][0] * sample_rate)]
                    # Audio lenght must not exceed 12 seconds
                    if chunk_audio.shape[1] > 12 * sample_rate or chunk_audio.shape[1] < 4 * sample_rate:
                        continue
                    torchaudio.save(wav_path, chunk_audio, sample_rate)
                except:
                    continue

                # Save xml
                try:
                    chunk.write('xml', fp=xml_path)
                except Exception as e:
                    continue

                # Save kern
                command = f'verovio -f musicxml-hum -t hum {xml_path} -o {kern_path} >/dev/null 2>&1'
                status = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if status.returncode != 0:
                    continue
                elif status.stderr:
                    if "Warning" in status.stderr:
                        continue
                try:
                    os.system(f'extractx -s 1 {kern_path} > temp/{split}/lower.krn')
                    os.system(f'extractx -s 2 {kern_path} > temp/{split}/upper.krn')
                    lower = Kern(Path(f'temp/{split}/lower.krn'))
                    upper = Kern(Path(f'temp/{split}/upper.krn'))
                    full = Kern(Path(kern_path))
                    flag = False
                    for kern in [lower, upper, full]:
                        succuess, cleaned = kern.clean()
                        if not succuess or cleaned:
                            flag = True
                            break
                    if flag: continue
                except Exception as e:
                    continue
                
                tiefix_status = False
                for j, kern in enumerate([lower, upper, full]):
                    subfolder = ['kern_lower', 'kern_upper', 'kern'][j]
                    kern_path = f'{feature_folder}/{subfolder}/{score_name}#{performance}.{i}.krn'
                    kern.save(Path(kern_path))
                    try:
                        # Fix ties with tiefix command
                        process = subprocess.run(['tiefix', kern_path],
                                                capture_output=True,
                                                encoding='iso-8859-1')
                        if (process.returncode != 0):
                            continue

                        kern = Kern(data=process.stdout)
                        kern.save(Path(kern_path))
                        if subfolder == 'kern':
                            tiefix_status = True
                    except Exception as e:
                        # logger.exception(f'Exception {e} while saving kern {chunk_path}')
                        continue   
                if not tiefix_status: continue
                lower = Kern(Path(lower_path))
                upper = Kern(Path(upper_path))

                # Save targets
                try:
                    lower = process_voices(lower)
                    upper = process_voices(upper)
                except Exception as e:
                    continue
                if lower is False or upper is False:
                    continue
                try:
                    lower = sort_voices(sort_chords(lower))
                    upper = sort_voices(sort_chords(upper))
                except:
                    continue
                if lower is False or upper is False:
                    continue

                lower = lower.tosequence()
                upper = upper.tosequence()
                if lower is None or upper is None:
                    continue
                try:
                    if lower.startswith('=\n'): lower = lower[2:]
                    if lower.endswith('\n='): lower = lower[:-2]
                    if upper.startswith('=\n'): upper = upper[2:]
                    if upper.endswith('\n='): upper = upper[:-2]
                    lower, upper = lower.split('\n=\n'), upper.split('\n=\n')
                    target = []
                    for m in range(5):
                        # Get key and time signature
                        current_key = int(downbeats[i+1+m][1])
                        current_time = downbeats[i+1+m][2]
                        if current_time not in self.time_sig_list:
                            target = []
                            break
                        target.append([current_key, current_time, labels.encode(lower[m]), labels.encode(upper[m])])
                    if len(target) != 5: continue
                    save(target, target_path)
                except Exception as e:
                    continue
        
        return unmatched

    def _get_score_name_from_folder(self, folder):
        folder = folder.split('/')
        folder = folder[folder.index('asap-dataset-master') + 1:]
        return '#'.join(folder)
    
    def _get_smallest_subdirectories(self):
        subdirectories = []

        def collect_subdirectories(path):
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

            if not subdirs and os.path.exists(os.path.join(path, 'xml_score.musicxml')):
                subdirectories.append(path)
            else:
                for subdir in subdirs:
                    collect_subdirectories(os.path.join(path, subdir))

        collect_subdirectories(self.asap_folder)
        return subdirectories

    def _get_anno_downbeats(self, anno_path):
        anno = load(anno_path)
        first_line = anno[0].split('\t')
        first_beat = first_line[2].split(',')
        upbeat = True if first_beat[0] == 'b' else False

        key, time_sig = None, None
        if len(first_beat) == 2:
            time_sig = first_beat[1]
        elif len(first_beat) == 3:
            time_sig = first_beat[1] if len(first_beat[1]) != 0 else time_sig
            key = first_beat[2] if len(first_beat[2]) != 0 else key
        downbeats = [(0, key, time_sig)]
        for line in anno:
            line = line.split('\t')
            time = float(line[0])
            beat = line[2].split(',')
            if len(beat) == 1:
                beat_type = beat[0]
            elif len(beat) == 2:
                beat_type = beat[0]
                time_sig = beat[1] if len(beat[1]) != 0 else time_sig
            elif len(beat) == 3:
                beat_type = beat[0]
                time_sig = beat[1] if len(beat[1]) != 0 else time_sig
                key = beat[2] if len(beat[2]) != 0 else key
            if beat_type == 'db':
                downbeats.append((time, key, time_sig))
        return upbeat, downbeats
    
    def _prepare_spectrograms(self):
        for split in ['train', 'test']:
            folder = os.path.join(self.feature_folder, f'{split}')
            if not os.path.exists(folder): continue
            print(f'Now processing: {split}')
            spectrogram_folder = os.path.join(folder, 'spectrogram')
            mkdirs(spectrogram_folder)
            target_files = os.listdir(os.path.join(folder, 'target'))
            for target_file in tqdm(target_files):
                name = target_file[:-4]
                wav_path = os.path.join(folder, 'wav', f'{name}.wav')
                spectrogram_path = os.path.join(spectrogram_folder, f'{name}.npy')
                if os.path.exists(spectrogram_path):
                    continue
                # Get wav duration
                waveform, sample_rate = torchaudio.load(wav_path)
                duration = waveform.shape[1] / sample_rate
                if duration > self.hparams['max_duration']:
                    continue
                # Get spectrogram
                spectrogram = get_VQT(wav_path, self.hparams["VQT_params"])
                save(spectrogram, spectrogram_path)

class ASAPDataset(Dataset):
    def __init__(self, hparams, split, device):
        """Dataset from ASAP database.

        Args:
            hparams: (types.SimpleNamespace) hyperparameters.
            split: (str) split of the dataset.
            """
        super().__init__()
        self.hparams = hparams
        self.split = split
        self.device = device
        self.feature_folder = os.path.join(hparams["feature_folder"], f'{self.split}')
        self.song_list = self.get_song_list()
        self.time_sig_list = load('data_processing/metadata/time_signature_list.json')
        self.time_sig_dict = {time_sig: i for i, time_sig in enumerate(self.time_sig_list)}

    def __len__(self):
        return len(self.song_list)

    def __getitem__(self, idx):
        song = self.song_list[idx]

        # get spectrogram
        spectrogram_file = os.path.join(self.feature_folder, 'spectrogram', f'{song}.npy')
        spectrogram = self.pad_spectrogram(load(spectrogram_file))  # (1, T, n_mels)

        # get score
        score_file = os.path.join(self.feature_folder, 'target', f'{song}.pkl')
        score = load(score_file)
        
        # Get key and time signature
        key_signatures = self.key_to_int([int(item[0]) for item in score])
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
               song, \
               'asap'

    def get_song_list(self):
        song_list = os.listdir(os.path.join(self.hparams["feature_folder"], f'{self.split}', 'spectrogram'))
        song_list = [song[:-4] for song in song_list]
        return song_list

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
        padded_measure = torch.ones((max_length,), dtype=torch.long) * labels.labels_map['<pad>']
        if len(measure) > max_length:
            measure = measure[:max_length]
        padded_measure[:len(measure)] = torch.tensor(measure)
        if len(measure) < max_length:
            padded_measure[len(measure)] = labels.labels_map['<eos>']
        return padded_measure.to(self.device) # (max_length,)

    def prepare_spectrograms(self):
        """Prepare the spectrograms for the dataset."""
        # create spectrogram folder
        feature_folder = os.path.join(self.hparams["feature_folder"], f'{self.split}')
        spectrogram_folder = os.path.join(feature_folder, 'spectrogram')
        mkdirs(spectrogram_folder)
        for i, wav in enumerate(os.listdir(os.path.join(feature_folder, 'wav'))):
            if i == 50:
                break
            print(f'Preparing spectrogram {i+1}/{len(self)}', end='\r')
            wav_file = os.path.join(feature_folder, 'wav', wav)
            song = wav[:-4]
            spectrogram_file = os.path.join(spectrogram_folder, f'{song}.npy')
            if os.path.exists(spectrogram_file):
                continue
            spectrogram = get_VQT(wav_file, self.hparams["VQT_params"])
            save(spectrogram, spectrogram_file)
        self.song_list = os.listdir(os.path.join(feature_folder, 'spectrogram'))
        self.song_list = [song[:-4] for song in self.song_list]

    def get_smallest_subdirectories(self, folder_path):
        subdirectories = []

        def collect_subdirectories(path):
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

            if not subdirs and os.path.exists(os.path.join(path, 'xml_score.musicxml')):
                subdirectories.append(path)
            else:
                for subdir in subdirs:
                    collect_subdirectories(os.path.join(path, subdir))

        collect_subdirectories(folder_path)
        return subdirectories

def unpad(full_seq):
    # full_seq: (max_length)
    length = (full_seq == EOS).nonzero()
    length = length[0][0] if length.shape[0] > 0 else full_seq.shape[0]
    return full_seq[:length].cpu().numpy()



if __name__ == '__main__':
    hparams = load('hparams/finetune.yaml')
    process = ProcessASAP(hparams)
    process.process_all()