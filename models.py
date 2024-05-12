import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from data_processing.humdrum import LabelsMultiple

labels = LabelsMultiple(extended=True)
SOS = labels.labels_map['<sos>']
EOS = labels.labels_map['<eos>']
vocab_size = len(labels.labels_map)

class ScoreTranscription(nn.Module):
    def __init__(self, in_channels=1, freq_bins=480, conv_feature_size=256, 
                 hidden_size=256, max_bars=5, num_time_sig=7, num_keys=14, 
                 max_length=(437, 129), note_emb_size=16, staff_emb_size=32, 
                 time_sig_emb_size=5, key_emb_size=8):
        super().__init__()
        self.convstack = ConvStack(in_channels, freq_bins, conv_feature_size)
        self.encoder = Encoder(conv_feature_size, hidden_size)
        self.decoder = HierarchicalDecoder(max_bars, num_time_sig, num_keys, hidden_size, 
                                           max_length, note_emb_size, staff_emb_size, 
                                           time_sig_emb_size, key_emb_size)

    def forward(self, 
                spectrogram, 
                inference=True,
                ground_truth=None, 
                teacher_forcing_ratio=0., 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device

        ## Score
        conv_outputs = self.convstack(spectrogram) # (B, max_length, hidden_size)
        
        # Encode
        encoder_outputs, hidden = self.encoder(conv_outputs)
            # encoder_outputs: (B, max_length, hidden_size*2), hidden: (1, B, hidden_size*2)
        # encoder_outputs = torch.cat([encoder_outputs, reg_onset_output.detach()], dim=-1) # (B, max_length, hidden_size*2)

        # Decode
        time_sig_outs, key_outs, upper_outs, lower_outs = \
            self.decoder(encoder_outputs, 
                         hidden, 
                         inference, 
                         ground_truth, 
                         teacher_forcing_ratio, 
                         device)

        return time_sig_outs, key_outs, upper_outs, lower_outs

class Encoder(nn.Module):
    """Encoder to encode spectrogram into latent vector.

        Args:
            input_size: size of the input feature.
            hidden_size: size of the hidden layer.
        """
    def __init__(self, input_size=200, hidden_size=200):
        super().__init__()
        
        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=2, 
                          batch_first=True, 
                          bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.init_weight()

    def init_weight(self):
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, x):
        # x: (B, max_length, input_size)
        x, hidden = self.gru(x)  # x: (B, max_length, hidden_size*2), hidden: (4, B, hidden_size)
        hidden1 = F.tanh(self.fc(torch.cat((hidden[0], hidden[1]), dim=1)))  # (B, hidden_size)
        hidden2 = F.tanh(self.fc(torch.cat((hidden[2], hidden[3]), dim=1)))  # (B, hidden_size)
        hidden = torch.cat((hidden1.unsqueeze(0), hidden2.unsqueeze(0)), dim=0)  # (2, B, hidden_size)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(0)  # (1, B, hidden_size*2)
        return x, hidden # x: (B, max_length, hidden_size*2), hidden: (1, B, hidden_size*2)

class HierarchicalDecoder(nn.Module):
    def __init__(self, max_bars=8, num_time_sig=9, num_keys=14, 
                 hidden_size=200, max_length=(437,129), note_emb_size=16, 
                 staff_emb_size=16, time_sig_emb_size=8, key_emb_size=8):
        super(HierarchicalDecoder, self).__init__()

        # Sizes
        self.max_bars = max_bars
        self.num_time_sig = num_time_sig
        self.time_sig_SOS = num_time_sig
        self.num_keys = num_keys
        self.key_SOS = num_keys
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.note_emb_size = note_emb_size
        self.staff_emb_size = staff_emb_size
        self.time_sig_emb_size = time_sig_emb_size
        self.key_emb_size = key_emb_size
        
        # Embedding layers
        self.note_emb = nn.Embedding(vocab_size, note_emb_size)
        self.time_sig_emb = nn.Embedding(num_time_sig + 1, time_sig_emb_size) # +1 for SOS
        self.key_emb = nn.Embedding(num_keys + 1, key_emb_size) 
        self.staff_emb = nn.GRU(note_emb_size,
                                staff_emb_size,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

        # Decoder layers
        self.upper_decoder = NoteDecoder(max_length[0], note_emb_size, hidden_size)
        self.lower_decoder = NoteDecoder(max_length[1], note_emb_size, hidden_size)
        self.attn = AttentionLayer(hidden_size)
        self.gru = nn.GRU(staff_emb_size*4 + time_sig_emb_size + key_emb_size + hidden_size*2,
                          hidden_size*2,
                          num_layers=1, 
                          batch_first=True)
        
        # Output linear layers
        self.time_sig_out = nn.Sequential(nn.Linear(hidden_size*4, hidden_size*4),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size*4, hidden_size*2),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size*2, num_time_sig))
        self.key_out = nn.Sequential(nn.Linear(hidden_size*4, hidden_size*4),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size*4, hidden_size*2),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size*2, num_keys))

        self.init_weight()
    
    def init_weight(self):
        init_gru(self.gru)
        # init_layer(self.time_sig_out)
        # init_layer(self.key_out)
    
    def get_SOS_token(self, batch_size):
        # Initialize bar tokens
        SOS_token = torch.ones(batch_size, 1, dtype=torch.long) * SOS              # (B, 1)
        SOS_token = self.note_emb(SOS_token.to(self.device))                       # (B, 1, note_emb_size)
        EOS_token = torch.ones(batch_size, 1, dtype=torch.long) * EOS              # (B, 1)
        EOS_token = self.note_emb(EOS_token.to(self.device))                       # (B, 1, note_emb_size)
        staff_token = torch.cat([SOS_token, EOS_token], dim=1)                     # (B, 2, note_emb_size)
        staff_token = self.staff_emb(staff_token)[-1].transpose(0, 1).contiguous() # (B, 2, staff_emb_size)
        staff_token = staff_token.view(-1, 2 * self.staff_emb_size).unsqueeze(1)   # (B, 1, staff_emb_size*2)
        bar_token = torch.cat([staff_token, staff_token], dim=-1)                  # (B, 1, staff_emb_size*4)

        # Initialize time signature tokens
        time_sig_token = torch.ones(batch_size, 1, dtype=torch.long) * self.time_sig_SOS # (B, 1)
        time_sig_token = self.time_sig_emb(time_sig_token.to(self.device))               # (B, 1, time_sig_emb_size)

        # Initialize key tokens
        key_token = torch.ones(batch_size, 1, dtype=torch.long) * self.key_SOS # (B, 1)
        key_token = self.key_emb(key_token.to(self.device))                    # (B, 1, key_emb_size)

        token = torch.cat([bar_token, time_sig_token, key_token], dim=-1)
            # (B, 1, staff_emb_size*4 + time_sig_emb_size + key_emb_size)
        return token, staff_token
    
    def get_staff_token_from_probs(self, score_probs, lengths):
        # score_probs: (B, max_steps, vocab_size)
        # lengths: (B)
        staff_token = torch.argmax(score_probs, dim=-1) # (B, max_steps)
        staff_token = self.note_emb(staff_token.to(self.device)) # (B, max_steps, note_emb_size)
        staff_token = pack_padded_sequence(staff_token, 
                                           lengths.cpu(), 
                                           batch_first=True, 
                                           enforce_sorted=False) # (B, max_steps, note_emb_size)
        staff_token = self.staff_emb(staff_token)[-1].transpose(0, 1).contiguous() # (B, 2, staff_emb_size)
        staff_token = staff_token.view(-1, 2 * self.staff_emb_size).unsqueeze(1) # (B, 1, staff_emb_size*2)

        return staff_token
    
    def get_staff_token_from_gt(self, score_gt, lengths):
        # score_gt: (B, max_steps)
        # lengths: (B)
        staff_token = self.note_emb(score_gt.to(self.device)) # (B, max_steps, note_emb_size)
        staff_token = pack_padded_sequence(staff_token,
                                           lengths.cpu(),
                                           batch_first=True,
                                           enforce_sorted=False) # (B, max_steps, note_emb_size)
        staff_token = self.staff_emb(staff_token)[-1].transpose(0, 1).contiguous() # (B, 2, staff_emb_size)
        staff_token = staff_token.view(-1, 2 * self.staff_emb_size).unsqueeze(1) # (B, 1, staff_emb_size*2)

        return staff_token
        
    def decode_bars(self, 
                    encoder_outputs, 
                    hidden, 
                    inference=True, 
                    ground_truth=None, 
                    teacher_forcing_ratio=0):
        # encoder_outputs: (B, max_length, hidden_size*2)
        # hidden: (1, B, hidden_size*2)
        # ground_truth: [time_sig, key, upper_score, lower_score]
        batch_size = encoder_outputs.shape[0]

        if inference:
            assert teacher_forcing_ratio == 0
            assert ground_truth is None
        
        if ground_truth is not None:
            time_sig_gt, key_gt, upper_score_gt, upper_lengths_gt, lower_score_gt, lower_lengths_gt = ground_truth
            # time_sig_gt: (B, max_bars)
            # key_gt: (B, max_bars)
            # upper_score_gt: (B, max_bars, max_steps)
            # upper_lengths_gt: (B, max_bars)
            # lower_score_gt: (B, max_bars, max_steps)
            # lower_lengths_gt: (B, max_bars)

        # Initialize SOS tokens
        token, _ = self.get_SOS_token(batch_size) # (B, 1, staff_emb_size*4 + time_sig_emb_size + key_emb_size)
        time_sig_token = torch.ones(batch_size, 1, dtype=torch.long) * self.time_sig_SOS # (B, 1)
        time_sig_token = self.time_sig_emb(time_sig_token.to(self.device)) # (B, 1, time_sig_emb_size)
        key_token = torch.ones(batch_size, 1, dtype=torch.long) * self.key_SOS # (B, 1)
        key_token = self.key_emb(key_token.to(self.device)) # (B, 1, key_emb_size)

        # Initialize outputs
        time_sig_outs = torch.zeros(batch_size, 
                                    self.max_bars, 
                                    self.num_time_sig).to(self.device) # (B, max_bars, num_time_sig)
        key_outs = torch.zeros(batch_size, 
                               self.max_bars, 
                               self.num_keys).to(self.device) # (B, max_bars, num_keys)
        upper_outs = torch.zeros(batch_size, 
                                 self.max_bars, 
                                 self.max_length[0], 
                                 vocab_size).to(self.device) # (B, max_bars, max_steps, vocab_size)
        lower_outs = torch.zeros(batch_size, 
                                 self.max_bars, 
                                 self.max_length[1], 
                                 vocab_size).to(self.device) # (B, max_bars, max_steps, vocab_size)

        for bar_index in range(self.max_bars):
            token = F.dropout(token, p=0.1, training=self.training, inplace=False)
            # Attention
            attn_weights = self.attn(hidden, encoder_outputs).unsqueeze(1) # (B, 1, max_length)
            context = torch.bmm(attn_weights, encoder_outputs) # (B, 1, hidden_size*2)

            # Bar-level GRU
            rnn_input = torch.cat([token, context], dim=2)
                # (B, 1, staff_emb_size*4 + time_sig_emb_size + key_emb_size + hidden_size*2)
            bar_summary, hidden = self.gru(rnn_input, hidden)
                # bar_summary: (B, 1, hidden_size*2), hidden: (1, B, hidden_size*2)
            
            # Decode notes
            if ground_truth is not None:
                input_gt_upper = upper_score_gt[:, bar_index, :] # (B, max_steps)
                input_gt_lower = lower_score_gt[:, bar_index, :] # (B, max_steps)
                input_tf_ratio = teacher_forcing_ratio
            else:
                input_gt_upper = None
                input_gt_lower = None
                input_tf_ratio = 0.
            
            # Decode upper staff
            upper_score_probs, upper_lengths = \
                self.upper_decoder(encoder_outputs, 
                                   bar_summary.transpose(0, 1), # (1, B, hidden_size*2)
                                   inference, 
                                   input_gt_upper, 
                                   input_tf_ratio,
                                   self.device) # (B, max_steps, vocab_size), (B)
            # Decode lower staff
            lower_score_probs, lower_lengths = \
                self.lower_decoder(encoder_outputs, 
                                   bar_summary.transpose(0, 1), # (1, B, hidden_size*2)
                                   inference, 
                                   input_gt_lower, 
                                   input_tf_ratio,
                                   self.device) # (B, max_steps, vocab_size)
            
            upper_outs[:, bar_index, :, :] = upper_score_probs # (B, max_steps, vocab_size)
            lower_outs[:, bar_index, :, :] = lower_score_probs # (B, max_steps, vocab_size)
            
            # Decode time signature and key
            time_input = torch.cat([bar_summary.squeeze(1), 
                                    context.squeeze(1)], dim=1) # (B, hidden_size*4)
            time_sig_outs[:, bar_index, :] = F.log_softmax(self.time_sig_out(time_input), dim=-1) # (B, num_time_sig)
            key_input = torch.cat([bar_summary.squeeze(1), 
                                   context.squeeze(1)], dim=1) # (B, hidden_size*4)
            key_outs[:, bar_index, :] = F.log_softmax(self.key_out(key_input), dim=-1) # (B, num_keys)
            
            # Update token
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force and not inference:
                upper_staff_token = \
                    self.get_staff_token_from_gt(upper_score_gt[:, bar_index, :], 
                                                 upper_lengths_gt[:, bar_index]) # (B, 1, staff_emb_size*2)
                lower_staff_token = \
                    self.get_staff_token_from_gt(lower_score_gt[:, bar_index, :], 
                                                 lower_lengths_gt[:, bar_index]) # (B, 1, staff_emb_size*2)
                staff_token = torch.cat([upper_staff_token, lower_staff_token], dim=-1)
                time_sig_token = self.time_sig_emb(time_sig_gt[:, bar_index]).unsqueeze(1) # (B, 1, time_sig_emb_size)
                key_token = self.key_emb(key_gt[:, bar_index]).unsqueeze(1) # (B, 1, key_emb_size)
                token = torch.cat([staff_token, time_sig_token, key_token], dim=-1)
                    # (B, 1, staff_emb_size*4 + time_sig_emb_size + key_emb_size)
            else:
                upper_staff_token = \
                    self.get_staff_token_from_probs(upper_score_probs, upper_lengths) # (B, 1, staff_emb_size*2)
                lower_staff_token = \
                    self.get_staff_token_from_probs(lower_score_probs, lower_lengths) # (B, 1, staff_emb_size*2)
                staff_token = torch.cat([upper_staff_token, lower_staff_token], dim=-1) # (B, 1, staff_emb_size*4)
                time_sig_token = self.time_sig_emb(torch.argmax(time_sig_outs[:, bar_index, :], dim=-1)).unsqueeze(1) # (B, 1, time_sig_emb_size)
                key_token = self.key_emb(torch.argmax(key_outs[:, bar_index, :], dim=-1)).unsqueeze(1) # (B, 1, key_emb_size)
                token = torch.cat([staff_token, time_sig_token, key_token], dim=-1)
                    # (B, 1, staff_emb_size*4 + time_sig_emb_size + key_emb_size)
        
        return (time_sig_outs,   # (B, max_bars, num_time_sig)
                key_outs,        # (B, max_bars, num_keys)
                upper_outs,      # (B, max_bars, max_steps, vocab_size)
                lower_outs,)     # (B, max_bars, max_steps, vocab_size)

    def forward(self, 
                encoder_outputs, 
                hidden, 
                inference=True, 
                ground_truth=None, 
                teacher_forcing_ratio=0,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        if inference:
            assert teacher_forcing_ratio == 0
            assert ground_truth is None
            time_sig_outs, key_outs, upper_outs, lower_outs = \
                self.decode_bars(encoder_outputs, hidden, True, None, 0.)
        else:
            time_sig_outs, key_outs, upper_outs, lower_outs = \
                self.decode_bars(encoder_outputs, hidden, False, ground_truth, teacher_forcing_ratio)

        return (time_sig_outs,  # (B, max_bars, num_time_sig)
                key_outs,       # (B, max_bars, num_keys)
                upper_outs,     # (B, max_bars, max_steps, vocab_size)
                lower_outs,)    # (B, max_bars, max_steps, vocab_size)

class NoteDecoder(nn.Module):
    def __init__(self, max_steps=25, note_emb_size=128, hidden_size=400):
        super(NoteDecoder, self).__init__()
        # decoding steps
        self.max_steps = max_steps
        
        # embedding sizes
        self.note_emb_size = note_emb_size
        self.hidden_size = hidden_size

        # Decoder layers
        self.embedding = nn.Embedding(vocab_size, note_emb_size)
        self.attn = AttentionLayer(hidden_size)
        self.gru = nn.GRU(note_emb_size + hidden_size*2,
                          hidden_size*2,
                          num_layers=1, 
                          batch_first=True)

        # output linear layers
        self.out = nn.Linear(hidden_size*4, vocab_size)
        self.init_weight()

    def init_weight(self):
        init_gru(self.gru)
        init_layer(self.out)

    def decode_notes(self, 
                     encoder_outputs, 
                     hidden, 
                     inference=True, 
                     ground_truth=None, 
                     teacher_forcing_ratio=0):
        # encoder_outputs: (B, max_length, hidden_size*2)
        # hidden: (1, B, hidden_size*2)
        # ground truth: (B, max_steps)
        if inference:
            assert teacher_forcing_ratio == 0
            assert ground_truth is None

        # initialize SOS token
        batch_size = encoder_outputs.shape[0]
        token = torch.ones(batch_size, 1, dtype=torch.long) * SOS # (B, 1)
        token = self.embedding(token.to(self.device)) # (B, 1, note_emb_size)

        # initialize outputs
        score_probs = torch.zeros(batch_size, self.max_steps, vocab_size).to(self.device) # (B, max_steps, vocab_size)
        EOS_labels = torch.zeros(batch_size)
        lengths = torch.zeros(batch_size, dtype=torch.long).fill_(self.max_steps) # (B)
        for t in range(0, self.max_steps):
            if EOS_labels.sum() == batch_size:
                break
            token = F.dropout(token, p=0.1, training=self.training, inplace=False)
            # attention
            attn_weights = self.attn(hidden, encoder_outputs).unsqueeze(1) # (B, 1, max_length)
            context = torch.bmm(attn_weights, encoder_outputs) # (B, 1, hidden_size*2)

            # GRU
            rnn_input = torch.cat([token, context], dim=2)  # (B, 1, note_emb_size + hidden_size*2)
            output, hidden = self.gru(rnn_input, hidden) # output: (B, 1, hidden_size*2), hidden: (1, B, hidden_size*2)
            output = torch.cat([output, context], dim=-1) # (B, 1, hidden_size*4)
            output = self.out(output) # (B, 1, vocab_size)
            prob = F.log_softmax(output, dim=-1) # (B, 1, vocab_size)
            score_probs[:, t, :] = prob.squeeze(1) # (B, vocab_size)

            teacher_force = random.random() < teacher_forcing_ratio
            if not inference and teacher_force:
                token = self.embedding(ground_truth[:, t].unsqueeze(1)) # (B, 1, note_emb_size)
            else:
                token = self.embedding(torch.argmax(prob, dim=-1)) # (B, 1, note_emb_size)
            
            # check EOS and add label
            for batch_index in range(batch_size):
                if ground_truth is not None:
                    if ground_truth[batch_index, t] == EOS:
                        EOS_labels[batch_index] = 1
                        lengths[batch_index] = t + 1
                else:
                    if torch.argmax(prob, dim=-1)[batch_index] == EOS:
                        EOS_labels[batch_index] = 1
                        lengths[batch_index] = t + 1
        return score_probs, lengths # (B, max_steps, vocab_size), (B)
    
    def forward(self, 
                encoder_outputs, 
                hidden, 
                inference=True, 
                ground_truth=None, 
                teacher_forcing_ratio=0, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        
        if inference:
            assert teacher_forcing_ratio == 0
            assert ground_truth is None
            score_probs, lengths = self.decode_notes(encoder_outputs, hidden, True, None, 0.)
        
        else:
            score_probs, lengths = self.decode_notes(encoder_outputs, hidden, False, ground_truth, teacher_forcing_ratio)
        return score_probs, lengths # (B, max_steps, vocab_size), (B)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.attn = nn.Linear(hidden_size*4, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.attn)
        init_layer(self.v)
        
    def forward(self, hidden, encoder_output):
        # encoder_outputs: (B, max_length, hidden_size*2)
        # hidden: (1, B, hidden_size*2)
        
        max_length = encoder_output.shape[1]
        hidden = hidden.transpose(0, 1).repeat(1, max_length, 1)  # (B, max_length, hidden_size*2)
        energy = F.tanh(self.attn(torch.cat((hidden, encoder_output), dim=2)))  # (B, max_length, hidden_size)
        attention = self.v(energy).squeeze(2)  # (B, max_length)
        
        return F.softmax(attention, dim=1)

class ConvStack(nn.Module):
    """Convolutional stack.

        Args:
            in_channels: number of channels in the input feature.
            freq_bins: number of frequency bins in the input feature.
            output_size: output size of the convolutional stack.
        """

    def __init__(self, in_channels=1, freq_bins=480, output_size=200):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=20, 
                               kernel_size=(3, 3), 
                               stride=(1, 1), 
                               padding=(1, 1), 
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=20, 
                               out_channels=20, 
                               kernel_size=(3, 3), 
                               stride=(1, 1), 
                               padding=(1, 1), 
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=20, 
                               out_channels=40, 
                               kernel_size=(3, 3), 
                               stride=(1, 1), 
                               padding=(1, 1), 
                               bias=False)
        self.conv4 = nn.Conv2d(in_channels=40, 
                               out_channels=40, 
                               kernel_size=(3, 3), 
                               stride=(1, 1), 
                               padding=(1, 1), 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(40)
        self.bn4 = nn.BatchNorm2d(40)
        
        self.out = nn.Linear(freq_bins*40, output_size, bias=False)
        self.out_bn = nn.BatchNorm1d(output_size)

        self.init_weight()
    
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        
        init_layer(self.out)
        init_bn(self.out_bn)
        
    def forward(self, x):
        # x: (B, 1, spectrogram_max_length, freq_bins)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.dropout(x, p=0.2, training=self.training)
        
        # x1 = F.avg_pool2d(x, kernel_size=(2, 1))
        # x2 = F.max_pool2d(x, kernel_size=(2, 1))
        # x = x1 + x2
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x = F.dropout(x, p=0.2, training=self.training)
                    # (B, 40, spectrogram_max_length/2, freq_bins)
        x = x.transpose(1, 2).flatten(2)
                    # (B, spectrogram_max_length/2, freq_bins*40)
        x = F.relu(self.out_bn(self.out(x).transpose(1, 2)).transpose(1, 2))
                    # (B, spectrogram_max_length/2, output_size)
        x = F.dropout(x, p=0.2, training=self.training)
        
        return x

## Model initialisation functions
## Code borrowed from https://github.com/bytedance/piano_transcription/blob/master/pytorch/models.py .

def init_layer(layer):
    """Initialise a linear or convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialise a batch norm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def init_gru(rnn):
    """Initialize a GRU layer. """
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


if __name__ == "__main__":
    device = "cpu"
    model = ScoreTranscription().to(device)
    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    spectrogram = torch.randn(1, 1, 1200, 480).to(device)
    for i in range(100):
        time_sig_outs, key_outs, upper_outs, lower_outs = model(spectrogram, device=device)
        print("Shape of time signature predictions: ", time_sig_outs.shape)
        print("Shape of key predictions: ", key_outs.shape)
        print("Shape of upper staff predictions: ", upper_outs.shape)
        print("Shape of lower staff predictions: ", lower_outs.shape)
        break