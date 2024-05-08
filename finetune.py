#!/usr/bin/env/python3
"""
Author: Zeng Wei 2024

Modified from recipe for training a sequence-to-sequence ASR system with librispeech in SpeechBrain.
"""

import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from datasets.asap import ASAPDataset
from utilities import save, load, mkdirs
from data_processing.humdrum import LabelsMultiple
from jiwer import wer
import os
import numpy as np
from sklearn.metrics import f1_score

labels = LabelsMultiple(extended=True)
SOS = labels.labels_map['<sos>']
EOS = labels.labels_map['<eos>']

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        spectrogram, time_sig_target, key_target, upper_target, upper_lengths, \
            lower_target, lower_lengths, song_name, version = batch
        # spectrogram: (B, 1, 800, 480)
        # upper_target: (B, max_length_upper)
        # lower_target: (B, max_length_lower)
        ground_truth = [time_sig_target, key_target, upper_target, upper_lengths, lower_target, lower_lengths]
        if stage == sb.Stage.TRAIN:
            time_sig_outs, key_outs, upper_outs, lower_outs = \
                self.modules.transcription(spectrogram=spectrogram,
                                           inference=False,
                                           ground_truth=ground_truth, 
                                           teacher_forcing_ratio=self.hparams.teacher_forcing_ratio,
                                           device=self.device)
        else:
            time_sig_outs, key_outs, upper_outs, lower_outs = \
                self.modules.transcription(spectrogram=spectrogram,
                                           inference=True,
                                           ground_truth=None, 
                                           teacher_forcing_ratio=0.,
                                           device=self.device)
        return time_sig_outs, key_outs, upper_outs, lower_outs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        spectrogram, time_sig_target, key_target, upper_target, upper_lengths, \
            lower_target, lower_lengths, song_name, version = batch
        time_sig_outs, key_outs, upper_outs, lower_outs = predictions

        loss = 0.

        # Calculate loss for time signature
        time_loss = self.hparams.loss_time_sig(time_sig_outs.permute(0,2,1), time_sig_target)
        
        # Calculate loss for key signature
        key_loss = self.hparams.loss_key(key_outs.permute(0,2,1), key_target)
        
        # Calculate loss for upper staff
        upper_outs_reshaped = upper_outs.view(upper_outs.shape[0] * upper_outs.shape[1], -1, upper_outs.shape[3])
        upper_target_reshaped = upper_target.view(upper_target.shape[0] * upper_target.shape[1], -1)
        upper_loss = self.hparams.loss_score(upper_outs_reshaped.permute(0,2,1), upper_target_reshaped)

        # Calculate loss for lower staff
        lower_outs_reshaped = lower_outs.view(lower_outs.shape[0] * lower_outs.shape[1], -1, lower_outs.shape[3])
        lower_target_reshaped = lower_target.view(lower_target.shape[0] * lower_target.shape[1], -1)
        lower_loss = self.hparams.loss_score(lower_outs_reshaped.permute(0,2,1), lower_target_reshaped)

        loss += time_loss + key_loss + upper_loss + lower_loss

        self.time_losses.append(time_loss.detach().cpu().numpy())
        self.key_losses.append(key_loss.detach().cpu().numpy())
        self.upper_losses.append(upper_loss.detach().cpu().numpy())
        self.lower_losses.append(lower_loss.detach().cpu().numpy())

        if stage != sb.Stage.TRAIN:
            # Record the predictions and targets from validation data
            for b in range(len(song_name)):
                id = song_name[b]
                # Upper staff
                pred = upper_outs[b].argmax(dim=-1)
                target = upper_target[b]
                self.upper_pred[id] = [unpad(p).tolist() for p in pred]
                self.upper_target[id] = [unpad(t).tolist() for t in target]
                
                # Lower staff
                pred = lower_outs[b].argmax(dim=-1)
                target = lower_target[b]
                self.lower_pred[id] = [unpad(p).tolist() for p in pred]
                self.lower_target[id] = [unpad(t).tolist() for t in target]

                # Key signature
                self.key_pred[id] = [key_out.argmax(dim=-1).item() for key_out in key_outs[b]]
                self.key_target[id] = [key.item() for key in key_target[b]]
                
                # Time signature
                self.time_sig_pred[id] = [time_sig_out.argmax(dim=-1).item() for time_sig_out in time_sig_outs[b]]
                self.time_sig_target[id] = [time_sig.item() for time_sig in time_sig_target[b]]
        
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.time_losses, self.key_losses, self.upper_losses, self.lower_losses = [], [], [], []
        if stage != sb.Stage.TRAIN:
            self.upper_pred, self.upper_target = {}, {}
            self.lower_pred, self.lower_target = {}, {}
            self.key_pred, self.key_target = {}, {}
            self.time_sig_pred, self.time_sig_target = {}, {}
            mkdirs(os.path.join(self.hparams.output_folder, 'results', 'valid'))
            mkdirs(os.path.join(self.hparams.output_folder, 'results', 'test'))
        self.time_sig_list = load('data_processing/metadata/time_signature_list.json')

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        stage_stats = {"loss": stage_loss,
                       "time_loss": np.mean(self.time_losses),
                       "key_loss": np.mean(self.key_losses),
                       "upper_loss": np.mean(self.upper_losses),
                       "lower_loss": np.mean(self.lower_losses)}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            if not hasattr(self, 'train_stats'):
                self.train_stats = {"loss": -1}
            wer_upper, wer_upper_dict = calculate_wer(self.upper_pred, self.upper_target)
            wer_lower, wer_lower_dict = calculate_wer(self.lower_pred, self.lower_target)
            key_f1, key_f1_dict = caculate_f1(self.key_pred, self.key_target)
            time_f1, time_f1_dict = caculate_f1(self.time_sig_pred, self.time_sig_target)
            stage_stats["key_f1"] = key_f1
            stage_stats["time_f1"] = time_f1
            # merge_songs(self.pred_seq, self.target_seq)
            stage_stats["WER_upper"] = wer_upper
            stage_stats["WER_lower"] = wer_lower
            stage_stats["WER"] = (wer_upper + wer_lower) / 2
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"], "WER": stage_stats["WER"]}, min_keys=["WER"],
            )

            # Save the predictions and targets
            for id in self.upper_pred:
                pred = []
                for i in range(len(self.upper_pred[id])):
                    key_pred = self.key_pred[id][i] - 6
                    time_sig_pred = self.time_sig_list[self.time_sig_pred[id][i]]
                    lower_pred = self.lower_pred[id][i]
                    upper_pred = self.upper_pred[id][i]
                    pred.append([key_pred, time_sig_pred, lower_pred, upper_pred])
                wer_upper = wer_upper_dict[id]
                wer_lower = wer_lower_dict[id]
                key_f1 = key_f1_dict[id]
                time_f1 = time_f1_dict[id]
                split = 'test' if stage == sb.Stage.TEST else 'valid'
                target_path = os.path.join(self.hparams.feature_folder, 'test', 'target', f'{id}.pkl')
                result_path = os.path.join(self.hparams.output_folder, 'results', split, f'{id}.json')
                result = {'target_path': target_path, 
                          'pred': pred, 'wer_upper': wer_upper, 'wer_lower': wer_lower, 
                          'key_f1': key_f1, 'time_f1': time_f1}
                save(result, result_path)

def calculate_wer(pred_seq, target_seq):
    wer_dict = {}
    n, total_wer = 0, 0
    for id in pred_seq:
        pred = [idx2string(p) for p in pred_seq[id]]
        target = [idx2string(t) for t in target_seq[id]]
        pred = " \n = \n ".join(pred)
        target = " \n = \n ".join(target)
        wer_dict[id] = wer(target, pred)
        total_wer += wer_dict[id]
        n += 1
    return total_wer / n, wer_dict

def idx2string(idx_seq):
    """Convert a batch of index matrix to a sequence of string."""
    seq = []
    for idx in idx_seq:
        seq.append(labels.labels_map_inv[idx])
    return ' '.join(seq)

def caculate_f1(pred, target):
    f1_dict = {}
    n, total_f1 = 0, 0
    for id in pred:
        f1_dict[id] = f1_score(target[id], pred[id], average='macro')
        total_f1 += f1_dict[id]
        n += 1
    return total_f1 / n, f1_dict

def unpad(full_seq):
    # full_seq: (max_length)
    length = (full_seq == EOS).nonzero()
    length = length[0][0] if length.shape[0] > 0 else full_seq.shape[0]
    return full_seq[:length].cpu().numpy()

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    pretrained_output_folder = hparams["pretrained_output_folder"]
    output_folder = hparams["output_folder"]
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Copy the pretrained model to the output folder
    os.system(f"cp -r {pretrained_output_folder}/save {output_folder}")

    # Dataset prep
    train_dataset = ASAPDataset(hparams, 'train', run_opts["device"])
    valid_dataset = ASAPDataset(hparams, 'test', run_opts["device"])
    test_dataset = ASAPDataset(hparams, 'test', run_opts["device"])
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_dataset,
        valid_dataset,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    import os

    # Testing
    asr_brain.evaluate(
        test_dataset,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        min_key="WER",
    )