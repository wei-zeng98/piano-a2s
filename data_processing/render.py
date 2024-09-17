import os
import sys
sys.path.append('.')
import random
import numpy as np
from tqdm import tqdm
from pedalboard import *
from midi2audio import FluidSynth
import soundfile as sf
import pyloudnorm as pyln
import warnings
import music21 as m21
import pretty_midi as pm
import pandas as pd
import multiprocessing
from pathlib import Path
from functools import partial
import logging
import subprocess
import torchaudio
from utilities import set_seed, load, save, mkdirs, get_VQT, MIDIProcess
from data_processing.humdrum import Kern, LabelsMultiple, sort_chords, sort_voices, process_voices

warnings.filterwarnings("ignore")
set_seed(0)

feasible_trasposes = {-6: [0, '-m2', '-m3', 'M2', 'M3'],                             # exclude '-M2', '-M3', 'm2', 'm3'
                      -5: [0, '-m2', '-m3', 'M2', 'M3'],                             # exclude '-M2', '-M3', 'm2', 'm3'
                      -4: [0, '-m2', '-M2', '-m3', 'M2', 'M3'],                      # exclude '-M3', 'm2', 'm3'
                      -3: [0, '-m2', '-M2', '-m3', 'M2', 'm3', 'M3'],                # exclude '-M3' 'm2'
                      -2: [0, '-m2', '-M2', '-m3', '-M3', 'M2', 'm3', 'M3'],         # exclude 'm2'
                      -1: [0, '-m2', '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],   # exclude nothing
                       0: [0, '-m2', '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],   # exclude nothing
                       1: [0, '-m2', '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],   # exclude nothing
                       2: [0, '-m2', '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],   # exclude nothing
                       3: [0, '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],          # exclude '-m2'
                       4: [0, '-M2', '-m3', '-M3', 'm2', 'M2', 'm3'],                # exclude '-m2', 'M3'
                       5: [0, '-M2', '-M3', 'm2', 'M2', 'm3'],                       # exclude '-m2', '-m3', 'M3'
                       6: [0, '-M2', '-M3', 'm2', 'm3'],                             # exclude '-m2', '-m3', 'M2', 'M3'
                       7: [0, '-M2', '-M3', 'm2', 'm3'],                             # exclude '-m2', '-m3', 'M2', 'M3'
                     }

def get_staff_spines(kern_path):
    kern = Kern(Path(kern_path))
    for line in kern.header:
        if line.startswith('**'):
            spines = line.split('\t')
            break
    indices = [i for i, x in enumerate(spines) if x == '**kern']
    return indices[0] + 1, indices[1] + 1

def split_single_score(score_path,
                       feature_folder,
                       labels,
                       time_sig_list,
                       logger,
                       split='train',
                       version=0,
                       chunk_size=5):
    # Get score name
    score_name = score_path.split('/')[-1].split('.')[0]

    # Make directory
    output_dir = os.path.join(feature_folder, f'{split}/{version}')
    mkdirs(output_dir)
    mkdirs(f'temp/{split}/{version}')
    for dir in ['midi', 'wav', 'kern', 'xml', 'target', 'kern_upper', 'kern_lower', 'info']:
        mkdirs(f'{output_dir}/{dir}')
    
    # Split into staffs and clean
    try:
        spine_lower, spine_upper = get_staff_spines(score_path)
        os.system(f'extractx -s {spine_lower} {score_path} > temp/{split}/{version}/lower.krn')
        os.system(f'extractx -s {spine_upper} {score_path} > temp/{split}/{version}/upper.krn')
        lower = Kern(Path(f'temp/{split}/{version}/lower.krn'))
        upper = Kern(Path(f'temp/{split}/{version}/upper.krn'))
        full = Kern(Path(score_path))
        for kern in [lower, upper, full]:
            cleaned, _ = kern.clean()
            if not cleaned:
                # logger.error(f'Cannot clean kern {score_path}')
                return
    except Exception as e:
        # logger.exception(f"Exception while cleaning {score_path} audio. Reason: {e}")
        return

    # Split into chunks
    chunks = []
    for i, kern in enumerate([lower, upper, full]):
        try:
            stride = 2 if split == 'train' else chunk_size
            kern_chunks = kern.split(chunk_size, stride)
        except Exception as e:
            # logger.exception(f'Exception {e} while splitting {score_path}')
            return
        
        for j, chunk in enumerate(kern_chunks):
            # Save kern
            subfolder = ['kern_lower', 'kern_upper', 'kern'][i]
            chunk_path = f'{output_dir}/{subfolder}/{score_name}.{j}.krn'
            chunk.save(Path(chunk_path))
            try:
                # Fix ties with tiefix command
                process = subprocess.run(['tiefix', chunk_path],
                                         capture_output=True,
                                         encoding='iso-8859-1')
                if (process.returncode != 0):
                    # logger.error(
                    #     f"tiefix error={process.returncode} on {chunk_path}")
                    # logger.error(process.stdout)
                    continue

                chunk = Kern(data=process.stdout)
                chunk.save(Path(chunk_path))
                if i == 2:
                    chunks.append(f'{score_name}.{j}.krn')
            except Exception as e:
                # logger.exception(f'Exception {e} while saving kern {chunk_path}')
                continue
    
    for chunk in chunks:
        if not os.path.exists(f'{output_dir}/kern_lower/{chunk}') \
            or not os.path.exists(f'{output_dir}/kern_upper/{chunk}'):
            # logger.error(f'Cannot find kern_lower or kern_upper {chunk}')
            continue

        # Information
        info_path = f'{output_dir}/info/{chunk[:-4]}.json'
        info = {'score_name': score_name, 'chunk': chunk}
        
        # Read as music21 score
        xml_path = f'{output_dir}/xml/{chunk[:-4]}.xml'
        try:
            os.system(f'hum2xml {output_dir}/kern/{chunk} >{xml_path}')
            m21_score = m21.converter.parse(xml_path).expandRepeats()
        except Exception as e:
            # logger.exception(f'Exception {e} while parsing {chunk}')
            continue

        if len(m21_score.parts[0].getElementsByClass('Measure')) != chunk_size:
            # logger.error(f'Wrong number of measures in {chunk}')
            continue

        # Transpose
        try:
            original_key = m21_score.parts[0].getElementsByClass('Measure')[0].keySignature.sharps
            if split == 'train':
                # Transpose to random key with feasible transpose
                transpose = random.choice(feasible_trasposes[original_key])
                info['original_key'] = original_key
                info['transpose'] = transpose
                m21_score = m21_score.transpose(transpose)
                p_transpose_lower = \
                    subprocess.run(['transpose', '-t', transpose, f'{output_dir}/kern_lower/{chunk}'],
                                    capture_output=True,
                                    encoding='iso-8859-1')
                p_transpose_upper = \
                    subprocess.run(['transpose', '-t', transpose, f'{output_dir}/kern_upper/{chunk}'],
                                    capture_output=True,
                                    encoding='iso-8859-1')
                lower = Kern(data=p_transpose_lower.stdout)
                upper = Kern(data=p_transpose_upper.stdout)
                lower.save(Path(f'{output_dir}/kern_lower/{chunk}'))
                upper.save(Path(f'{output_dir}/kern_upper/{chunk}'))
            else:
                info['original_key'] = original_key
                info['transpose'] = 0
                lower = Kern(Path(f'{output_dir}/kern_lower/{chunk}'))
                upper = Kern(Path(f'{output_dir}/kern_upper/{chunk}'))
        except:
            continue
        
        # Save transposed xml
        try:
            m21_score.write('musicxml', fp=xml_path)
        except Exception as e:
            # logger.exception(f'Exception {e} while saving xml {xml_path}')
            continue
        
        # Save targets
        try:
            lower = process_voices(lower)
            upper = process_voices(upper)
        except:
            # logger.error(f'Cannot process voices in {chunk}')
            continue
        if lower is False or upper is False:
            # logger.error(f'Cannot process voices in {chunk}')
            continue
        try:
            lower = sort_voices(sort_chords(lower))
            upper = sort_voices(sort_chords(upper))
        except:
            # logger.error(f'Cannot sort voices in {chunk}')
            continue
        if lower is False or upper is False:
            # logger.error(f'Cannot sort voices in {chunk}')
            continue
        lower = lower.tosequence()
        upper = upper.tosequence()
        if lower is None or upper is None:
            # logger.error(f'Double sharps/flats/dots found in {chunk}')
            continue
        current_key, current_time = None, None
        try:
            target_path = f'{output_dir}/target/{chunk[:-4]}.pkl'
            if lower.startswith('=\n'): lower = lower[2:]
            if lower.endswith('\n='): lower = lower[:-2]
            if upper.startswith('=\n'): upper = upper[2:]
            if upper.endswith('\n='): upper = upper[:-2]
            lower, upper = lower.split('\n=\n'), upper.split('\n=\n')
            target = []
            for m in range(chunk_size):
                # Get key and time signature
                key_signature = m21_score.parts[0].getElementsByClass('Measure')[m].keySignature
                time_signature = m21_score.parts[0].getElementsByClass('Measure')[m].timeSignature
                current_key = key_signature.sharps if key_signature is not None else current_key
                current_time = time_signature.ratioString if time_signature is not None else current_time
                if current_time not in time_sig_list:
                    # logger.error(f'Invalid time signature {current_time} in {chunk}')
                    target = []
                    break
                if current_key < -6 or current_key > 7:
                    # logger.error(f'Invalid key signature {current_key} in {chunk}')
                    target = []
                    break
                target.append([current_key, current_time, labels.encode(lower[m]), labels.encode(upper[m])])
            if len(target) != chunk_size: continue
            save(target, target_path)
            save(info, info_path)
        except Exception as e:
            # logger.exception(f'Exception {e} while saving target {target_path}')
            continue

def split_datasets(versions, feature_folder):
    logger = logging.getLogger('data_prep')
    
    # Get score paths
    score_paths = []
    for kern_file in os.listdir('data_processing/kern'):
        score_paths.append(os.path.join('data_processing/kern', kern_file))
    print(f'Number of scores: {len(score_paths)}')

    # Get test and validation split
    test_songs = set([row['name'] for i, row in 
                      pd.read_csv('data_processing/metadata/test_split.txt').iterrows()])
    val_songs = set([row['name'] for i, row in 
                     pd.read_csv('data_processing/metadata/valid_split.txt').iterrows()])

    labels = LabelsMultiple(extended=True)
    time_sig_list = load('data_processing/metadata/time_signature_list.json')
    
    # Split scores
    for v in versions:
        print(f'Version {v}')
        for i, score_path in tqdm(enumerate(score_paths), total=len(score_paths)):
            score_name = score_path.split('/')[-1].split('.')[0]
            if score_name in test_songs and v == 0:
                split = 'test'
            elif score_name in val_songs and v == 0:
                split = 'valid'
            elif score_name not in test_songs and score_name not in val_songs:
                split = 'train'
            else:
                continue
            split_single_score(score_path, 
                               feature_folder, 
                               labels, 
                               time_sig_list, 
                               logger=logger, 
                               split=split, 
                               version=v)

def render_all_midi(versions, feature_folder, soundfont_folder):
    train_sondfonts = ['TimGM6mb.sf2', 
                       'FluidR3_GM.sf2', 
                       'UprightPianoKW-20220221.sf2', 
                       'SalamanderGrandPiano-V3+20200602.sf2']
    test_soundfonts = ['UprightPianoKW-20220221.sf2', 
                       'SalamanderGrandPiano-V3+20200602.sf2', 
                       'YDP-GrandPiano-20160804.sf2']
    dynamic_compression = Compressor(threshold_db=-1, ratio = 18, attack_ms=50)

    for split in ['train', 'valid']:
        folder = os.path.join(feature_folder, f'{split}')
        for v in versions:
            current_folder = os.path.join(folder, str(v))
            print(f'Now processing: {split}, {v}')
            if not os.path.exists(os.path.join(current_folder, 'midi')): continue
            midi_files = os.listdir(os.path.join(current_folder, 'midi'))
            for midi_file in tqdm(midi_files):
                if split == 'train':
                    soundfont = random.choice(train_sondfonts)
                else:
                    soundfont = random.choice(test_soundfonts)
                midi_path = os.path.join(current_folder, 'midi', midi_file)
                wav_path = os.path.join(current_folder, 'wav', midi_file[:-4] + f'~{soundfont[:-4]}.wav')
                soundfont_path = os.path.join(soundfont_folder, soundfont)
                render_one_midi(FluidSynth(soundfont_path, sample_rate=44100),
                                dynamic_compression,
                                midi_path,
                                wav_path)
    
    for split in ['test']:
        folder = os.path.join(feature_folder, f'{split}')
        for v in versions:
            current_folder = os.path.join(folder, str(v))
            print(f'Now processing: {split}, {v}')
            if not os.path.exists(os.path.join(current_folder, 'midi')): continue
            midi_files = os.listdir(os.path.join(current_folder, 'midi'))
            for midi_file in tqdm(midi_files):
                for soundfont in test_soundfonts:
                    midi_path = os.path.join(current_folder, 'midi', midi_file)
                    wav_path = os.path.join(current_folder, 'wav', midi_file[:-4] + f'~{soundfont[:-4]}.wav')
                    soundfont_path = os.path.join(soundfont_folder, soundfont)
                    render_one_midi(FluidSynth(soundfont_path, sample_rate=44100),
                                    dynamic_compression,
                                    midi_path,
                                    wav_path)

def render_one_midi(fs, dynamic_compression, midi_path, wav_path):
    # fs: FluidSynth object
    try:
        # print('Now processing: ', midi_path)
        fs.midi_to_audio(midi_path, wav_path)
        data, rate = sf.read(wav_path)  
        meter = pyln.Meter(rate) # Create BS.1770 meter
        if np.ndim(data) > 1:
            data = np.mean(data, axis=1) # Convert to mono
        
        data_copy = pyln.normalize.peak(data, -1.0)
        attempt = 0
        while meter.integrated_loudness(data_copy) < -20:
            loudness_normalized_audio = pyln.normalize.peak(data, -1.0)
            threshold = meter.integrated_loudness(loudness_normalized_audio) + 15
            if attempt % 3 == 2:
                dynamic_compression.threshold_db -= 1
                if dynamic_compression.threshold_db < threshold: break
            elif attempt % 3 == 1:
                dynamic_compression.attack_ms *= 0.7
                if dynamic_compression.attack_ms < 3: break
            else:
                dynamic_compression.ratio += 2
                if dynamic_compression.ratio > 34: break
            loudness_normalized_audio = np.array(loudness_normalized_audio)
            data_copy = dynamic_compression(loudness_normalized_audio, rate)
            data_copy = pyln.normalize.peak(data_copy, -1.0)
            attempt += 1

        dynamic_compression.threshold_db = -5
        dynamic_compression.attack_ms = 10
        dynamic_compression.ratio = 1
        attempt = 0

        data = data_copy
        data_copy = pyln.normalize.loudness(data, meter.integrated_loudness(data), -15)
    
        while data_copy.max() > 0.9 or data_copy.min() < -0.9:
            data_copy = pyln.normalize.loudness(data, meter.integrated_loudness(data), -15)
            if attempt % 3 == 2:
                dynamic_compression.threshold_db -= 0.5
                if dynamic_compression.threshold_db < -10: break
            elif attempt % 3 == 1:
                dynamic_compression.attack_ms *= 0.75
                if dynamic_compression.attack_ms < 1: break
            else:
                dynamic_compression.ratio += 1.5
                if dynamic_compression.ratio > 15: break
            loudness_normalized_audio = np.array(data_copy)
            data_copy = dynamic_compression(loudness_normalized_audio, rate)
            attempt += 1

        dynamic_compression.threshold_db = -1
        dynamic_compression.attack_ms = 50
        dynamic_compression.ratio = 18
        attempt = 0

        data = pyln.normalize.peak(data_copy, -1.0)

        sf.write(wav_path, data, rate)
    except ValueError:
        print(wav_path)
        with open('errors.txt','a') as f:
            f.write(wav_path)
            f.write('\n')

def xml_to_midi(versions, feature_folder, midi_syn='epr'):
    assert midi_syn in ['epr', 'score']
    train_composers = ['score', 'Bach', 'Balakirev', 'Beethoven',
                       'Brahms', 'Debussy', 'Glinka', 'Haydn',
                       'Liszt', 'Prokofiev', 'Rachmaninoff',
                       'Ravel', 'Schubert', 'Schumann', 'Scriabin']
    test_composers = ['score', 'Bach', 'Mozart', 'Chopin']
    
    if midi_syn == 'epr':
        # Load virtuosoNet and work under virtuosoNet folder
        os.chdir('virtuosoNet')
        sys.path.append(os.getcwd())
        from virtuosoNet.model_run import load_file_and_generate_performance
        # We use 4 composers for test set, with each version corresponding to a composer
        for v in range(1, 4):
            os.system(f'cp -r {feature_folder}/valid/0/ {feature_folder}/valid/{v}')
            os.system(f'cp -r {feature_folder}/test/0/ {feature_folder}/test/{v}')
        
    for split in ['train', 'test', 'valid']:
        for v in versions:
            if split != 'train':
                if midi_syn == 'epr' and v >= 4: continue
                if midi_syn == 'score' and v > 0: continue
            print(f'Now processing: {split}, {v}')
            folder = os.path.join(feature_folder, f'{split}', str(v))
            mkdirs(f'temp/{split}/{v}')
            target_files = os.listdir(os.path.join(folder, 'target'))
            file_names = [file[:-4] for file in target_files]
            for file_name in tqdm(file_names):
                xml_path = os.path.join(folder, 'xml', f'{file_name}.xml')
                midi_path = os.path.join(folder, 'midi', f'{file_name}.mid')
                info_path = os.path.join(folder, 'info', f'{file_name}.json')
                info = load(info_path)
                os.system(f'cp {xml_path} temp/{split}/{v}/xml.xml')
                if split == 'train':
                    composer = random.choice(train_composers) if midi_syn == 'epr' else 'score'
                else:
                    composer = test_composers[v] if midi_syn == 'epr' else 'score'
                info['composer'] = composer
                try:
                    if composer == 'score':
                        command = f'verovio -f musicxml-hum -t midi {xml_path} -o temp/{split}/{v}/temp.mid'
                        status = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        if status.returncode != 0:
                            print(f'Error: {xml_path}')
                            continue
                        elif status.stderr:
                            if 'Warning' in status.stderr or 'Error' in status.stderr:
                                print(f'{xml_path}: {status.stderr}')
                                continue
                        midiprocess = MIDIProcess(f'temp/{split}/{v}/temp.mid', split)
                    else:
                        # EPR
                        load_file_and_generate_performance(path_name=f'temp/{split}/{v}/',
                                                        composer=composer)
                        midiprocess = MIDIProcess(f'test_result/{v}_by_isgn_z0.mid', split)
                    scaling, original_length = midiprocess.process(midi_path)
                    if scaling is not None:
                        info['scaling'] = scaling
                        info['original_length'] = original_length
                        save(info, info_path)
                except Exception as e:
                    print(f'Error: {xml_path}')
                    continue
    
    if midi_syn == 'epr':
        # Move back to the parent directory
        os.chdir('..')
    return

def convert_xml_to_kern(xml_folder='data_processing/xml'):
    print('Converting MuseSyn xml files to kern files...')
    xml_files = os.listdir(xml_folder)
    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(xml_folder, xml_file)
        kern_path = os.path.join('data_processing/kern', xml_file.replace('.xml', '.krn'))
        status = os.system(f'verovio -f musicxml-hum -t hum {xml_path} -o {kern_path} >/dev/null 2>&1')

def preprocess_kern():
    print('Preprocessing kern files...')
    mkdirs('data_processing/temp')
    kern_folder = 'data_processing/kern'
    kern_files = os.listdir(kern_folder)
    selected_chopin = set([row['name'] for i, row in pd.read_csv('data_processing/metadata/selected_chopin.txt').iterrows()])
    for kern_file in tqdm(kern_files):
        if kern_file.startswith('chopin'):
            if kern_file[:-4].split('#')[1] not in selected_chopin:
                os.remove(os.path.join(kern_folder, kern_file))
                continue
        elif kern_file.startswith('joplin'):
            kern_path = os.path.join(kern_folder, kern_file)
            if kern_file == 'joplin#school.krn':
                os.remove(kern_path)
                continue
            temp_xml_path = os.path.join('data_processing/temp', 'temp.xml')
            status = os.system(f'hum2xml {kern_path} >{temp_xml_path}')
            if status != 0:
                os.remove(kern_path)
                continue
            status = os.system(f'verovio -f musicxml-hum -t hum {temp_xml_path} -o {kern_path} >/dev/null 2>&1')
            if status != 0:
                os.remove(kern_path)
                continue

def prepare_spectrograms(versions, feature_folder, hparams):
    for split in ['train', 'valid', 'test']:
        for v in versions:
            folder = os.path.join(feature_folder, f'{split}', str(v))
            if not os.path.exists(folder): continue
            print(f'Now processing: {split}, {v}')
            spectrogram_folder = os.path.join(folder, 'spectrogram')
            mkdirs(spectrogram_folder)
            wav_files = os.listdir(os.path.join(folder, 'wav'))
            for wav_file in tqdm(wav_files):
                wav_path = os.path.join(folder, 'wav', wav_file)
                spectrogram_path = os.path.join(spectrogram_folder, wav_file[:-4] + '.npy')
                if os.path.exists(spectrogram_path):
                    continue
                # Get wav duration
                waveform, sample_rate = torchaudio.load(wav_path)
                duration = waveform.shape[1] / sample_rate
                if duration > hparams['max_duration']:
                    continue
                # Get spectrogram
                spectrogram = get_VQT(wav_path, hparams["VQT_params"])
                save(spectrogram, spectrogram_path)

def clean_files(versions, feature_folder):
    time_sig_list = load('data_processing/metadata/time_signature_list.json')
    for split in ['train', 'valid', 'test']:
        for v in versions:
            folder = os.path.join(feature_folder, f'{split}', str(v))
            if not os.path.exists(folder): continue
            midi_files = os.listdir(os.path.join(folder, 'midi'))
            deleted = 0
            for midi in tqdm(midi_files):
                name = midi[:-4]
                midi_path = f'{folder}/midi/{midi}'
                target_path = f'{folder}/target/{name}.pkl'
                if not os.path.exists(target_path):
                    os.remove(midi_path)
                    deleted += 1
                    continue

                # Check if midi is out of range
                midi_data = pm.PrettyMIDI(midi_path)
                duration = midi_data.get_end_time()
                if duration > 12:
                    os.remove(target_path)
                    os.remove(midi_path)
                    deleted += 1
                    continue

                flag_to_delete = False
                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        if note.pitch < 21 or note.pitch > 108:
                            os.remove(target_path)
                            os.remove(midi_path)
                            deleted += 1
                            flag_to_delete = True
                            break
                if flag_to_delete:
                    continue
                
                # Check if key or time signature is out of range
                target = load(target_path)
                for measure in target:
                    key = measure[0]
                    time = measure[1]
                    if key < -6 or key > 7 or time not in time_sig_list:
                        os.remove(target_path)
                        os.remove(midi_path)
                        deleted += 1
                        break
            print(f'{split}, {v}: {deleted} files deleted')

if __name__ == '__main__':
    # Make sure workspace is set correctly in hparams/pretrain.yaml before running
    hparams = load('hparams/pretrain.yaml')
    midi_syn = hparams['midi_syn'] # ['epr', 'score'] indicating how the midi files are synthesized
    feature_folder = hparams["feature_folder"]
    soundfont_folder = hparams["soundfont_folder"]

    # Convert MuseSyn XML files to Kern files
    convert_xml_to_kern()

    # Select certain Chopin scores, and reformat Joplin scores
    preprocess_kern()

    # Split scores into train, valid, test and cut into chunks
    print('Splitting scores into train, valid, test and cutting into chunks...')
    partial_work = partial(split_datasets, feature_folder=feature_folder)
    with multiprocessing.Pool(processes=5) as pool:
        versions_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        pool.map(partial_work, versions_list)

    # Convert xml to midi
    print('Converting xml to midi...')
    xml_to_midi(range(10), feature_folder=feature_folder, midi_syn=midi_syn)

    # Remove files with invalid key or time signature or length > 12s
    print('Cleaning files...')
    clean_files(range(10), feature_folder=feature_folder)

    # Synthesize midi files
    print('Synthesizing midi files...')
    partial_work = partial(render_all_midi, feature_folder=feature_folder, soundfont_folder=soundfont_folder)
    with multiprocessing.Pool(processes=5) as pool:
        versions_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        pool.map(partial_work, versions_list)

    # Prepare spectrograms
    print('Preparing spectrograms...')
    partial_work = partial(prepare_spectrograms, feature_folder=feature_folder, hparams=load('hparams/pretrain.yaml'))
    with multiprocessing.Pool(processes=5) as pool:
        versions_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        pool.map(partial_work, versions_list)
