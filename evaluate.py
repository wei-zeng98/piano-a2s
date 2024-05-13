import os
from tqdm import tqdm
import numpy as np
from utilities import load, save, mkdirs
from data_processing.humdrum import get_xml_from_target
from hyperpyyaml import load_hyperpyyaml
from subprocess import check_output
import subprocess

def get_mv2h_from_test(output_folder, split, mv2h_bin):
    results_dir = f'{output_folder}/results'
    mkdirs(f'{results_dir}/mv2h')
    for dir in ['scores', 'midi']:
        for sub_dir in ['pred', 'target']:
            mkdirs(f'{results_dir}/{dir}/{sub_dir}')
    
    errors = []
    for result in tqdm(os.listdir(f'{results_dir}/{split}')):
        id = result[:-5]
        pred_xml_path = f'{results_dir}/scores/pred/{id}_pred.xml'
        target_xml_path = f'{results_dir}/scores/target/{id}_target.xml'
        pred_midi_path = f'{results_dir}/midi/pred/{id}_pred.mid'
        target_midi_path = f'{results_dir}/midi/target/{id}_target.mid'
        mv2h_path = f'{results_dir}/mv2h/{id}_mv2h.json'
        if os.path.exists(mv2h_path): continue
        result = load(os.path.join(f'{results_dir}/{split}', result))

        # Convert xml to midi
        try:
            pred = get_xml_from_target(result['pred'])
            pred.write('musicxml', pred_xml_path)
            pred.write('midi', pred_midi_path)
            target = get_xml_from_target(load(result['target_path']))
            target.write('musicxml', target_xml_path)
            target.write('midi', target_midi_path)
        except Exception as e:
            errors.append(id)
            continue
        try:
            output = check_output(['sh', 
                                   'evaluate_midi_mv2h.sh', 
                                   target_midi_path, 
                                   pred_midi_path, 
                                   mv2h_bin], timeout=10)
        except ValueError as e:
            print('Failed to evaluate pair: \ntarget midi: {}\noutput midi: {}'.format(target_midi_path,
                                                                            pred_midi_path))
        except subprocess.TimeoutExpired as e:
            print('Timeout: \ntarget midi: {}\noutput midi: {}'.format(target_midi_path,
                                                                            pred_midi_path))
            continue
        
        result_list = output.decode('utf-8').splitlines()[-6:]
        result = dict([tuple(item.split(': ')) for item in result_list])
        for key, value in result.items():
            result[key] = float(value)
        if result['MV2H'] == 0:
            errors.append(id)
            continue # Errors in reading midi files
        save(result, mv2h_path)
    
    error_path = f'{results_dir}/errors.txt'
    with open(error_path, 'w') as f:
        for error in errors:
            f.write(error + '\n')

def summarize_syn_mv2h(results_dir, composer='all', soundfont='all', test_split='all'):
    assert composer in ['all', 'score', 'Bach', 'Mozart', 'Chopin']
    assert soundfont in ['all', 'Upright', 'Salamander', 'YDP']
    assert test_split in ['all', 'musesyn', 'humsyn']
    mv2h_folder = f'{results_dir}/results/mv2h'
    keys = ['Multi-pitch', 'Voice', 'Meter', 'Value', 'Harmony', 'MV2H']
    mv2h_metrics = {}
    for key in keys:
        mv2h_metrics[key] = 0
    n = 0
    for mv2h_file in tqdm(os.listdir(mv2h_folder)):
        id = mv2h_file[:-5]
        v, chunk_id, sf = id.split('~')

        # Only consider the specified composer and soundfont
        skip = False
        for i, c in enumerate(['score', 'Bach', 'Mozart', 'Chopin']):
            if composer == c and int(v) != i:
                skip = True
                break
        for s in ['Upright', 'Salamander', 'YDP']:
            if soundfont == s and sf[0] != s[0]:
                skip = True
                break
        if test_split == 'musesyn' and chunk_id[0].islower():
            skip = True
        if test_split == 'humsyn' and chunk_id[0].isupper():
            skip = True
        if skip: continue

        mv2h_path = os.path.join(mv2h_folder, mv2h_file)
        mv2h = load(mv2h_path)
        for key in keys:
            mv2h_metrics[key] += (mv2h[key] - mv2h_metrics[key]) / (n + 1)
        n += 1
    print(mv2h_metrics)
    print((mv2h_metrics['Multi-pitch'] + mv2h_metrics['Voice'] + mv2h_metrics['Value'] + mv2h_metrics['Harmony']) / 4)

def summarize_asap_mv2h(results_dir):
    mv2h_folder = f'{results_dir}/results.pretrain/mv2h'
    keys = ['Multi-pitch', 'Voice', 'Meter', 'Value', 'Harmony', 'MV2H']
    mv2h_metrics = {}
    for key in keys:
        mv2h_metrics[key] = 0
    n = 0
    for mv2h_file in tqdm(os.listdir(mv2h_folder)):
        mv2h_path = os.path.join(mv2h_folder, mv2h_file)
        mv2h = load(mv2h_path)
        for key in keys:
            mv2h_metrics[key] += (mv2h[key] - mv2h_metrics[key]) / (n + 1)
        n += 1
    print(mv2h_metrics)
    print((mv2h_metrics['Multi-pitch'] + mv2h_metrics['Voice'] + mv2h_metrics['Value'] + mv2h_metrics['Harmony']) / 4)

def summarize_WER_and_F1(results_dir):
    folder = f'{results_dir}/results/test'
    keys = ['wer_upper', 'wer_lower', 'key_f1', 'time_f1']
    metrics = {}
    for key in keys:
        metrics[key] = 0
    i = 0
    for result_file in tqdm(os.listdir(folder)):
        result_path = os.path.join(folder, result_file)
        result = load(result_path)
        for key in keys:
            metrics[key] += (result[key] - metrics[key]) / (i + 1)
        i += 1
    metrics['wer'] = (metrics['wer_upper'] + metrics['wer_lower']) / 2
    print(metrics)

def get_ER(results_dir):
    pred_scores_folder = f'{results_dir}/results/scores/pred'
    target_scores_folder = f'{results_dir}/results/scores/target'
    mv2h_folder = f'{results_dir}/results/mv2h'
    files = os.listdir(mv2h_folder)
    files = [file[:-10] for file in files if file.endswith('.json')]
    ers = np.zeros(11)
    i = 0
    for file in tqdm(files):
        try:
            pred_path = os.path.join(pred_scores_folder, file + '_pred')
            target_path = os.path.join(target_scores_folder, file + '_target')
            os.system(f'./MUSTER/evaluate_XML_voicePlus.sh {pred_path} {target_path} ER >/dev/null 2>&1')
            current_er = load('ER.txt')
            current_er = current_er[0].split(',')[12].split('\t')
            current_er = np.array([float(er) for er in current_er[1:]])
            if current_er.any() is np.nan: continue
        except Exception as e:
            continue
        for j in range(11):
            ers[j] += current_er[j]
            if ers.any() is np.nan: 
                print(ers)
                print(current_er)
        i += 1
    ers /= i
    print(ers)
    print(i)
    
    # Delete non-xml files
    for pred_file in os.listdir(pred_scores_folder):
        if not pred_file.endswith('.xml'):
            os.remove(os.path.join(pred_scores_folder, pred_file))
    for target_file in os.listdir(target_scores_folder):
        if not target_file.endswith('.xml'):
            os.remove(os.path.join(target_scores_folder, target_file))

if __name__ == '__main__':
    hparams = load_hyperpyyaml('hparams/finetune.yaml')
    pretrain_output_folder = hparams['pretrained_output_folder']
    finetune_output_folder = hparams['output_folder']
    mv2h_bin = hparams['mv2h_bin']

    # Get mv2h for test set
    get_mv2h_from_test(pretrain_output_folder, 'test', mv2h_bin)
    get_mv2h_from_test(finetune_output_folder, 'test', mv2h_bin)
    
    # Summarize mv2h
    summarize_syn_mv2h(pretrain_output_folder, composer='all', soundfont='all', test_split='all')
    summarize_asap_mv2h(finetune_output_folder)
