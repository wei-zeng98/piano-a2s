import os
import re
import numpy as np
import music21 as m21
from fractions import Fraction
from itertools import cycle
from pathlib import Path
import subprocess

classic_tempos = {
    "grave": 32,
    "largoassai": 40,
    "largo": 50,
    "pocolargo": 60,
    "adagio": 71,
    "pocoadagio": 76,
    "andante": 92,
    "andantino": 100,
    "menuetto": 112,
    "moderato": 114,
    "pocoallegretto": 116,
    "allegretto": 118,
    "allegromoderato": 120,
    "pocoallegro": 124,
    "allegro": 130,
    "moltoallegro": 134,
    "allegroassai": 138,
    "vivace": 140,
    "vivaceassai": 150,
    "allegrovivace": 160,
    "allegrovivaceassai": 170,
    "pocopresto": 180,
    "presto": 186,
    "prestoassai": 200,
}


class Labels(object):  # 38 symbols
    def __init__(self):
        # autopep8: off
        # yapf: disable
        self.labels = [
            "+",  # ctc blank
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "C", "D", "E", "F", "G", "A", "B", "c", "d", "e", "f", "g", "a", "b",  # noqa E501
            "r", "#", "-", "=", ".", "[", "_", "]", ";", "\t", "\n",
            "<", ">"  # seq2seq <sos> and <eos> delimiters
        ]
        # yapf: enable
        # autopep8: on
        self.labels_map = dict([(c, i) for (i, c) in enumerate(self.labels)])
        self.labels_map_inv = dict([(i, c)
                                    for (i, c) in enumerate(self.labels)])

    def ctclen(self, tokens):
        count = len(tokens)
        count += sum([tokens[i - 1] == tokens[i] for i in range(1, count)])
        return count

    def encode(self, chars):
        tokens = []
        for c in chars:
            tokens.append(self.labels_map[c])
        return tokens

    def decode(self, tokens):
        return list(filter(None, [self.labels_map_inv.get(t) for t in tokens]))


class LabelsMultiple(object):  # 148/173 symbols
    def __init__(self, extended=False):
        # yapf: disable
        # autopep8: off
        self.labels = [
            "1","1.","2","2.","4","4.","8","8.","16","16.","32","32.","64","64.","3","6","12","24","48","96",                           # noqa E501 E231
            "BBB#","CC","CC#","DD-","DD","DD#","EE-","EE","EE#","FF-","FF","FF#","GG-","GG","GG#","AA-","AA","AA#","BB-","BB","BB#",    # noqa E501 E231
            "C-","C","C#","D-","D","D#","E-","E","E#","F-","F","F#","G-","G","G#","A-","A","A#","B-","B","B#",                          # noqa E501 E231
            "c-","c","c#","d-","d","d#","e-","e","e#","f-","f","f#","g-","g","g#","a-","a","a#","b-","b","b#",                          # noqa E501 E231
            "cc-","cc","cc#","dd-","dd","dd#","ee-","ee","ee#","ff-","ff","ff#","gg-","gg","gg#","aa-","aa","aa#",                      # noqa E501 E231
            "bb-","bb","bb#",                                                                                                           # noqa E501 E231
            "ccc-","ccc","ccc#","ddd-","ddd","ddd#","eee-","eee","eee#","fff-","fff","fff#","ggg-","ggg","ggg#","aaa-","aaa","aaa#",    # noqa E501 E231
            "bbb-","bbb","bbb#",                                                                                                        # noqa E501 E231
            "cccc-","cccc","cccc#","dddd-","dddd","dddd#","eeee-","eeee","eeee#","ffff-","ffff",                                        # noqa E501 E231
            "r", ".", "[", "_", "]", ";", "\t", "\n", "<b>",
            "<sos>", "<eos>", "<pad>"  # seq2seq delimiters
        ]
        if extended:
            self.labels.extend([
                "128","20","40","176","112",                                                # noqa E501 E231
                "CCC","CCC#","DDD-","DDD","DDD#","EEE-","EEE","EEE#","FFF-","FFF",          # noqa E501 E231
                "FFF#","GGG-","GGG","GGG#","AAA-","AAA","AAA#","BBB-","BBB","CC-"           # noqa E501 E231
            ])
        # yapf: enable
        # autopep8: on
        self.labels_map = dict([(c, i) for (i, c) in enumerate(self.labels)])
        self.labels_map_inv = dict([(i, c)
                                    for (i, c) in enumerate(self.labels)])

    def encode(self, chars):
        tokens = []
        for line in chars.splitlines():
            chords = line.split('\t')
            for chord in chords:
                notes = chord.split(' ')
                for i, note in enumerate(notes):
                    if len(note) == 1:
                        tokens.append(self.labels_map[note])
                    else:
                        matchobj = re.fullmatch(
                            r'(\[?)(\d+\.*)([a-gA-Gr]{1,4}[\-#]*)(;?)([\]_]?)',
                            note)
                        if not matchobj:
                            raise Exception(
                                f'Item {note} in {line} does not match')
                        for m in [
                                matchobj[1], matchobj[2], matchobj[3], matchobj[4],
                                matchobj[5]
                        ]:
                            if m:
                                tokens.append(self.labels_map[m])
                    tokens.append(self.labels_map['<b>'])
                if tokens[-1] == self.labels_map['<b>']:
                    tokens.pop(-1)
                tokens.append(self.labels_map['\t'])
            tokens[-1] = self.labels_map['\n']
        tokens.pop(-1)
        return tokens

    def decode(self, tokens):
        decoded = list(filter(None, [self.labels_map_inv.get(t) for t in tokens]))
        return [i if i != '<b>' else ' ' for i in decoded]


class LabelsSingle(object):  # 9146/13631 symbols
    def __init__(self, extended=False):
        # yapf: disable
        # autopep8: off
        durations = ["1","1.","2","2.","4","4.","8","8.","16","16.","32","32.","64","64.","3","6","12","24","48","96"]                          # noqa E501 E231
        notes = ["BBB#","CC","CC#","DD-","DD","DD#","EE-","EE","EE#","FF-","FF","FF#","GG-","GG","GG#","AA-","AA","AA#","BB-","BB","BB#",       # noqa E501 E231
            "C-","C","C#","D-","D","D#","E-","E","E#","F-","F","F#","G-","G","G#","A-","A","A#","B-","B","B#",                                  # noqa E501 E231
            "c-","c","c#","d-","d","d#","e-","e","e#","f-","f","f#","g-","g","g#","a-","a","a#","b-","b","b#",                                  # noqa E501 E231
            "cc-","cc","cc#","dd-","dd","dd#","ee-","ee","ee#","ff-","ff","ff#","gg-","gg","gg#","aa-","aa","aa#",                              # noqa E501 E231
            "bb-","bb","bb#",                                                                                                                   # noqa E501 E231
            "ccc-","ccc","ccc#","ddd-","ddd","ddd#","eee-","eee","eee#","fff-","fff","fff#","ggg-","ggg","ggg#","aaa-","aaa","aaa#",            # noqa E501 E231
            "bbb-","bbb","bbb#",                                                                                                                # noqa E501 E231
            "cccc-","cccc","cccc#","dddd-","dddd","dddd#","eeee-","eeee","eeee#"]                                                               # noqa E501 E231
        if extended:
            durations.extend(["128","20","40","176","112"])                                     # noqa E501 E231
            notes.extend([
                "CCC","CCC#","DDD-","DDD","DDD#","EEE-","EEE","EEE#","FFF-","FFF","FFF#",       # noqa E501 E231
                "GGG-","GGG","GGG#","AAA-","AAA","AAA#","BBB-","BBB","CC-", "ffff-","ffff"])    # noqa E501 E231
        # yapf: enable
        # autopep8: on
        # ties = ["[", "_", "]"]
        self.labels = []
        for d in durations:
            for n in notes:
                self.labels.append(d + n)
                self.labels.append('[' + d + n)
                self.labels.append(d + n + '_')
                self.labels.append(d + n + ']')
            self.labels.append(d + 'r')
        self.labels.extend([
            ".",
            "\t",
            "\n",
            "<sos>",
            "<eos>",  # seq2seq delimiters
            "<pad>"
        ])
        self.labels_map = dict([(c, i) for (i, c) in enumerate(self.labels)])
        self.labels_map_inv = dict([(i, c)
                                    for (i, c) in enumerate(self.labels)])

    def encode(self, chars):
        tokens = []
        for line in chars.splitlines():
            items = line.split('\t')
            for item in items:
                tokens.append(self.labels_map[item])
                tokens.append(self.labels_map['\t'])
            tokens[-1] = self.labels_map['\n']
        tokens.pop(-1)
        return tokens

    def decode(self, tokens):
        return list(filter(None, [self.labels_map_inv.get(t) for t in tokens]))


class Humdrum(object):
    def __init__(self, path=None, data=None):
        if path:
            data = path.read_text(encoding='iso-8859-1')
        lines = data.splitlines()
        body_begin = 0
        body_end = 0
        for i, line in enumerate(lines):
            if line.startswith('**'):
                body_begin = i + 1
            if line.startswith('*-'):
                body_end = i
                break
        self.header = lines[:body_begin]
        self.footer = lines[body_end:]
        self.body = lines[body_begin:body_end]
        self.spine_types = self.header[-1].split('\t')

    def save(self, path):
        return path.write_text(self.dump(), encoding='iso-8859-1')

    def dump(self):
        return '\n'.join(self.header + self.body + self.footer)


class SpineInfo(object):
    def __init__(self, spine_types):
        self.spines = []
        for stype in spine_types:
            self.spines.append({
                'type': stype,
                'instrument': '*',
                'clef': '*',
                'keysig': '*',
                'tonality': '*',
                'timesig': '*',
                'metronome': '*',
            })

    def update(self, line):
        for i, item in enumerate(line.split('\t')):
            if item.startswith('*k['):
                self.spines[i]['keysig'] = item
            elif item.startswith('*clef'):
                self.spines[i]['clef'] = item
            elif item.startswith('*I'):
                self.spines[i]['instrument'] = item
            elif item.startswith('*MM'):
                self.spines[i]['metronome'] = item
            elif item.startswith('*M'):
                self.spines[i]['timesig'] = item
            elif item.startswith('*CT'):
                item = f'*MM{classic_tempos[item[3:]]}'
                self.spines[i]['metronome'] = item
            elif item.endswith(':'):
                self.spines[i]['tonality'] = item

    def override_instruments(self, instruments):
        pool = cycle(instruments)
        inst = instruments[0]
        for i in range(len(self.spines)):
            if self.spines[i]['type'] == '**kern':
                inst = next(pool)
            self.spines[i]['instrument'] = f'*I{inst}'

    def dump(self):
        header = []
        for v in [
                'type', 'instrument', 'clef', 'keysig', 'tonality', 'timesig',
                'metronome'
        ]:
            header.append('\t'.join([x[v] for x in self.spines]))
        footer = ['\t'.join(['*-' for x in self.spines])]
        return header, footer

    def clone(self):
        spine_types = [s['type'] for s in self.spines]
        spineinfo = SpineInfo(spine_types)
        spineinfo.spines = self.spines.copy()
        return spineinfo

class Kern(Humdrum):
    def __init__(self, path=None, data=None, constrained=False):
        super(Kern, self).__init__(path, data)

        self.constrained = constrained
        self.spines = SpineInfo(self.spine_types)
        self.first_line = 0
        for i, line in enumerate(self.body):
            if not line.startswith('*') or re.search(r'\*[\^v]', line):
                self.first_line = i
                break
            self.spines.update(line)

    def clean(self, remove_pauses=True):
        spine_types = self.spine_types.copy()
        base_spine_len = len(spine_types)
        newbody = []
        cleaned = False

        for line in self.body[self.first_line:]:
            if len(line) == 0:
                continue
            if re.search(r'\*[+x\^v]', line):
                i = 0
                remove_spine = False
                newline = []
                min_split_counts = 100
                for item in line.split('\t'):
                    if item.startswith(('*+', '*x')):
                        print('Unsupported variable spines')
                        return False, None
                    if item == '*^':
                        spine_types.insert(i + 1, f'{spine_types[i]}**split')
                        i += 1
                    elif item == '*v':
                        min_split_counts = min(min_split_counts,
                                               spine_types[i].count('**split'))
                        if remove_spine:
                            spine_types.pop(i)
                            i -= 1
                        else:
                            remove_spine = True
                    else:
                        if remove_spine:
                            # Last was removed:
                            # Transform first spine into simpler type.
                            spine_types[i - 1] = (
                                f"{spine_types[i-1].replace('**split', '')}"
                                f"{min_split_counts * '**split'}")
                        remove_spine = False
                    i += 1
                    newline.append(item)
                if not self.constrained:
                    newbody.append('\t'.join(newline))

                continue

            if line.startswith('!'):
                # Support for local comments (one per spine starting with '!').
                if self.constrained:
                    newline = []
                    items = line.split('\t')
                    for i, item in enumerate(items):
                        if spine_types[i].endswith(
                                '**split') and base_spine_len < len(items):
                            # Remove spline split
                            continue
                        newline.append(item)
                    newbody.append('\t'.join(newline))
                else:
                    newbody.append(line)
                continue

            # Remove unwanted symbols
            newline = []
            note_found = False
            grace_note_found = False

            items = line.split('\t')
            for i, item in enumerate(items):
                if self.constrained and spine_types[i].endswith('**split') \
                        and base_spine_len < len(items):
                    # print(f'Discarding item! {line}: {item}')
                    # Remove spline split
                    continue

                if spine_types[i].startswith('**kern') and \
                        not item.startswith(('*', '=')):
                    if self.constrained:
                        item = item.split()[0]  # Take first note of chord
                    if re.search(r'[pTtMmWwS$O:]', item):
                        # Remove ornaments
                        item = re.sub(r'[pTtMmWwS$O:]', r'', item)
                        cleaned = True
                    if remove_pauses:
                        item = re.sub(r';', r'', item)  # Remove pauses
                    item = re.sub(r'[JKkL\\/]', r'',
                                  item)  # Remove beaming and stems
                    item = re.sub(
                        r'[(){}xXyY&]', r'', item
                    )  # Remove slurs, phrases, elisions and editorial marks
                    item = re.sub(r'(\d*\.*r)(.*)', r'\1',
                                  item)  # Remove the rests line position.
                    if re.search('[qQP]', item):
                        grace_note_found = True
                        cleaned = True
                    elif re.search('[A-Ga-g]', item):
                        note_found = True
                newline.append(item)

            # Remove grace note lines unless they contain a non-grace note in the same time line
            if grace_note_found and not note_found:
                continue

            if grace_note_found and note_found:
                print(f'Unremovable grace notes {line}')
                return False, None

            if not all([x == '.' for x in newline]) and \
                    not all([x == '!' for x in newline]):
                newbody.append('\t'.join(newline))

        header, footer = self.spines.dump()
        self.body = header[1:] + newbody
        self.first_line = len(header) - 1
        return True, cleaned

    def split(self, chunk_size, stride=None):
        chunks = []
        spines = self.spines.clone()

        measures = [self.first_line]
        for i, line in enumerate(self.body[self.first_line:]):
            if re.match(r'^=(\d+|=)[^-]*', line):
                measures.append(i + self.first_line + 1)
        i = 0
        while i < len(measures) - 1:
            m_begin = measures[i]
            m_end = measures[i + chunk_size]

            header, footer = spines.dump()

            i += stride if stride else chunk_size

            final_measurement = False
            if len(measures) - i - 1 < chunk_size:
                body = self.body[m_begin:]
                final_measurement = True
            else:
                body = self.body[m_begin:m_end]

            if final_measurement:
                break

            # Fix spine splits
            if not self.constrained:
                # Fix first line of body.
                len_spines = len(self.spine_types)
                if len_spines != len(body[0].split('\t')):
                    # Get lines splits until len match spines, index is reversed
                    split_lines = []
                    lookup_body = self.body[:m_begin]

                    for line in lookup_body[::-1]:
                        # Instead of updating the spines, just insert all
                        # modifications after the original header
                        # if re.search(r'\*[\^v]', line):
                        if re.search(r'\*|:$', line):
                            split_lines.append(line)
                            if len(line.split('\t')) == len_spines:
                                break
                    # Insert all split lines in correct order:
                    for split_line in split_lines:
                        body.insert(0, split_line)

                # Fix footer. Skip comments.
                last = -1
                while body[last].startswith('!'):
                    last -= 1

                if len(footer[0].split('\t')) != len(body[last].split('\t')):
                    footer = [
                        '\t'.join(['*-' for x in body[last].split('\t')])
                    ]

            chunk = Kern(data='\n'.join(header + body + footer))
            chunks.append(chunk)

            # If not removing splits, no need to update the spines for the
            # next chunks as all split lines and marks are added after header.
            if self.constrained:
                for line in self.body[m_begin:measures[i]]:
                    if line.startswith('*'):
                        spines.update(line)

        return chunks

    def tosequence(self):
        spine_types = self.spine_types.copy()
        krn = []
        for line in self.body[self.first_line:]:
            newline = []
            if line.startswith('='):
                if not re.match(r'^=(\d+|=)[^-]*', line):
                    continue
                newline.append('=')
            elif not self.constrained and re.search(r'\*[\^v]', line):
                i = 0
                remove_spine = False
                min_split_counts = 100
                for item in line.split('\t'):
                    if item == '*^':
                        spine_types.insert(i + 1, f'{spine_types[i]}**split')
                        i += 1
                    elif item == '*v':
                        min_split_counts = min(min_split_counts,
                                               spine_types[i].count('**split'))
                        if remove_spine:
                            spine_types.pop(i)
                            i -= 1
                        else:
                            remove_spine = True
                    else:
                        if remove_spine:
                            # Last was removed:
                            # Transform first spine into simpler type.
                            spine_types[i - 1] = (
                                f"{spine_types[i-1].replace('**split', '')}"
                                f"{min_split_counts * '**split'}")
                        remove_spine = False
                    i += 1
                continue
            elif line.startswith(('*', '!')):
                continue
            else:
                line = re.sub(r'[^rA-Ga-g0-9.\[_\]#\-;\t ]', r'',
                              line)  # Remove undefined symbols
                for i, item in enumerate(line.split('\t')):
                    if spine_types[i].startswith('**kern'):
                        # Chords splitting:
                        if not self.constrained and ' ' in item:
                            chord = item.split()
                            newchord = []
                            for note in chord:
                                newchord.append(note)
                            newline.append(' '.join(newchord))
                        else:
                            newline.append(item)

            krn.append('\t'.join(newline))

        krnseq = '\n'.join(krn)

        if re.search(r'(#|-|\.){2,}', krnseq):
            # Discard double sharps/flats/dots
            return None

        return krnseq

def sort_voices(kern):
    n_voices = 1
    begin, end = 0, 0
    for i, line in enumerate(kern.body):
        if not (line.startswith('!') or line.startswith('!!')) and len(line.split('\t')) > 2: # voices more than 2
            return False
        if line.startswith('*^'):
            n_voices += 1
            begin = i + 1
            voice1 = []
            voice2 = []
        elif line.startswith('*v') or (n_voices == 2 and i == len(kern.body) - 1):
            n_voices -= 1
            end = i
            voice1 = np.mean([np.mean(i) for i in voice1 if len(i) > 0])
            voice2 = np.mean([np.mean(i) for i in voice2 if len(i) > 0])
            if voice1 < voice2:
                for j in range(begin, end):
                    if not (kern.body[j].startswith('!') or kern.body[j].startswith('!!')):
                        line = kern.body[j].split('\t')
                        if len(line) < 2:
                            continue
                        if len(line) > 2:
                            return False
                        line[0], line[1] = line[1], line[0]
                        kern.body[j] = '\t'.join(line)
        
        if n_voices == 2 and len(line.split('\t')) == 2:
            voice1.append(get_chords_pitches(line.split('\t')[0]))
            voice2.append(get_chords_pitches(line.split('\t')[1]))
    return kern

def sort_chords(kern):
    for i, line in enumerate(kern.body):
        if line.startswith('*') or line.startswith('!') or line.startswith('!!'):
            continue
        line = line.split('\t')
        sorted_line = []
        for chords in line:
            chords = chords.split(' ')
            if len(chords) == 1:
                sorted_line.append(chords[0])
                continue
            pitches = []
            for note in chords:
                matchobj = re.findall(
                            r'(?:[a-gA-G]{1,4}[\-#]*)',
                            note)
                if matchobj is not None and len(matchobj) > 0:
                    pitch = matchobj[0]
                    pitches.append(kern_to_midi(pitch))
            combined = list(zip(pitches, chords))
            sorted_combined = sorted(combined, key=lambda x: x[0])
            sorted_chords = [item[1] for item in sorted_combined]
            sorted_line.append(' '.join(sorted_chords))
        kern.body[i] = '\t'.join(sorted_line)
    return kern

def get_chords_pitches(chords):
    chords = chords.split(' ')
    pitches = []
    for note in chords:
        matchobj = re.findall(
                    r'(?:[a-gA-G]{1,4}[\-#]*)',
                    note)
        if matchobj is not None and len(matchobj) > 0:
            pitch = matchobj[0]
            pitches.append(kern_to_midi(pitch))
    return pitches

def kern_to_midi(kern_note):
    kern_to_midi_mapping = {
        'c': 60, 'd': 62, 'e': 64, 'f': 65, 'g': 67, 'a': 69, 'b': 71,
        'C': 48, 'D': 50, 'E': 52, 'F': 53, 'G': 55, 'A': 57, 'B': 59,
    }

    if kern_note[-1] == '#':
        midi_number = 1
        kern_note = kern_note[:-1]
    elif kern_note[-1] == '-':
        midi_number = -1
        kern_note = kern_note[:-1]
    else:
        midi_number = 0

    midi_number += kern_to_midi_mapping[kern_note[0]]

    if kern_note[0].isupper():
        midi_number -= 12 * (len(kern_note) - 1)
    else:
        midi_number += 12 * (len(kern_note) - 1)

    return midi_number

def get_n_measures_from_xml(xml_path):
    score = m21.converter.parse(xml_path).expandRepeats()
    return len(score.parts[0].getElementsByClass('Measure'))

def check_notes_existance(measure):
    for chord in measure:
        chord = chord.split(' ')
        for note in chord:
            matchobj = re.search(
                            r'(\[?)(\d+\.*)([a-gA-G]{1,4}[\-#]*)(;?)([\]_]?)',
                            note)
            if matchobj:
                return True
    return False

def check_single_voice(voice_l, voice_r):
    assert len(voice_l) == len(voice_r)
    note_duration_sets = [set(), set()]
    for i, voice in enumerate([voice_l, voice_r]):
        start_time = 0
        for chords in voice:
            note = chords.split(' ')[0]
            matchrest = re.search(r'(\[?)(\d+\.*)([r]{1,4}[\-#]*)', note)
            if matchrest:
                note_type = matchrest[2]
                if note_type.endswith('.'):
                    note_type = note_type[:-1]
                    start_time += Fraction(1, int(note_type)) + Fraction(1, 2 * int(note_type))
                else:
                    start_time += Fraction(1, int(note_type))
                continue
            matchobj = re.search(
                            r'(\[?)(\d+\.*)([a-gA-G]{1,4}[\-#]*)(;?)([\]_]?)',
                            note)
            if matchobj:
                note_type = matchobj[2]
                if note_type.endswith('.'):
                    note_type = note_type[:-1]
                    end_time = start_time + Fraction(1, int(note_type)) + Fraction(1, 2 * int(note_type))
                else:
                    end_time = start_time + Fraction(1, int(note_type))
                note_duration_sets[i].add((str(start_time), str(end_time)))
                start_time = end_time
                continue
    if note_duration_sets[1].issubset(note_duration_sets[0]):
        return 1
    elif note_duration_sets[0].issubset(note_duration_sets[1]):
        return 2
    else:
        return 0

def merge_voices(voice_l, voice_r):
    assert len(voice_l) == len(voice_r)
    length = len(voice_l)
    notes_existed_l = check_notes_existance(voice_l)
    notes_existed_r = check_notes_existance(voice_r)
    merged_voice = []
    n_voices = 1
    if notes_existed_l and notes_existed_r:
        # Check if the two voices could be merged into one
        single_voice = check_single_voice(voice_l, voice_r)
        if single_voice == 0:
            # Cannot merge into one voice
            for i in range(length):
                if voice_l[i] == 'null':
                    merged_voice.append(voice_r[i])
                elif voice_r[i] == 'null':
                    merged_voice.append(voice_l[i])
                else:
                    merged_voice.append(voice_l[i] + '\t' + voice_r[i])
            n_voices = 2
        elif single_voice == 1:
            # Merge into voice_l
            for i in range(length):
                if voice_l[i] == 'null':
                    merged_voice.append(voice_r[i])
                elif voice_r[i] == 'null':
                    merged_voice.append(voice_l[i])
                elif re.search(r'(\[?)(\d+\.*)([a-gA-G]{1,4}[\-#]*)(;?)([\]_]?)', voice_r[i]) is not None:
                    merged_voice.append(voice_l[i]+ ' ' + voice_r[i])
                else:
                    merged_voice.append(voice_l[i])
            n_voices = 1
        elif single_voice == 2:
            # Merge into voice_r
            for i in range(length):
                if voice_l[i] == 'null':
                    merged_voice.append(voice_r[i])
                elif voice_r[i] == 'null':
                    merged_voice.append(voice_l[i])
                elif re.search(r'(\[?)(\d+\.*)([a-gA-G]{1,4}[\-#]*)(;?)([\]_]?)', voice_l[i]) is not None:
                    merged_voice.append(voice_r[i]+ ' ' + voice_l[i])
                else:
                    merged_voice.append(voice_r[i])
            n_voices = 1
        
    elif not notes_existed_l and not notes_existed_r:
        if 'null' in voice_l:
            merged_voice = voice_r
        elif 'null' in voice_r:
            merged_voice = voice_l

    else:
        voice_to_keep = voice_l if notes_existed_l else voice_r
        voice_to_discard = voice_l if notes_existed_r else voice_r
        for i in range(length):
            if voice_to_keep[i] == 'null' or voice_to_keep[i] == '*' and voice_to_discard[i] != 'null':
                voice_to_keep[i] = voice_to_discard[i]
        merged_voice = voice_to_keep
    
    return merged_voice, n_voices

def merge_whole_chunk(voices, n_voices):
    assert len(voices) == len(n_voices)
    n_measure = len(voices)
    current_n_voices = 1
    result = []
    for i in range(n_measure):
        if current_n_voices == 1:
            result.append(f'={i+1}')
            if n_voices[i] == 1:
                result.extend(voices[i])
            elif n_voices[i] == 2:
                result.extend(voices[i])
                current_n_voices = 2
        elif current_n_voices == 2:
            if n_voices[i] == 1:
                result.append(f'={i+1}')
                result.extend(voices[i])
                current_n_voices = 1
            elif n_voices[i] == 2:
                result.append(f'={i+1}\t={i+1}')
                result.extend(voices[i])
    result.append('=')
    return add_split_token(result)

def add_split_token(body):
    added_result = []
    prev_n_voices = 1
    for line in body:
        if line.startswith('!'): continue
        current_n_voices = len(line.split('\t'))
        if current_n_voices == 2 and prev_n_voices == 1:
            added_result.append('*^')
        elif current_n_voices == 1 and prev_n_voices == 2:
            added_result.append('*v\t*v')
        added_result.append(line)
        prev_n_voices = current_n_voices
    return added_result

def process_voices(kern):
    """For two voices, merge them into one if possible."""
    i_measure = 0
    before_measure = []
    measure_voices = []
    measure_n_voices = []
    first_bar = False
    end_of_before_measure = False
    for i, line in enumerate(kern.body):
        if i_measure == 0:
            if kern.body[i+1].startswith('=') or line.startswith('*^'):
                end_of_before_measure = True
            if not end_of_before_measure:
                before_measure.append(line)

        if line.startswith('=') or (not first_bar and not kern.body[i+1].startswith('*') and not kern.body[i+1].startswith('!')):
            first_bar = True
            if i_measure != 0:
                # End of measure
                voice, n_voice = merge_voices(voice_l, voice_r)
                if len(voice) != 0:
                    measure_voices.append(voice)
                    measure_n_voices.append(n_voice)
            # New measure
            i_measure += 1
            voice_l, voice_r = [], []
            continue

        if line == '*^' or line == '*v\t*v':
            continue
        
        if i_measure != 0:
            line = line.split('\t')
            if len(line) == 1:
                voice_l.append(line[0])
                voice_r.append('null')
            elif len(line) == 2:
                voice_l.append(line[0])
                voice_r.append(line[1])
            elif len(line) > 2:
                return False
    measures = merge_whole_chunk(measure_voices, measure_n_voices)
    footer = kern.footer
    for i, line in enumerate(footer):
        footer[i] = line.split('\t')[0]
    return Kern(data='\n'.join(kern.header + before_measure + measures + footer))

def eliminate_duplicate_chords(kern):
    for i, line in enumerate(kern.body):
        if line.startswith('=') or line.startswith('*'):
            continue
        chords = line.split('\t')
        formatted_new_line = []
        for chord in chords:
            chord = chord.split(' ')
            if len(chord) > 1:
                chord = list(set(chord))
                # remove empty string
                chord = [x for x in chord if len(x) > 0]
                formatted_new_line.append(' '.join(chord))
            else:
                formatted_new_line.append(chord[0])
        
        kern.body[i] = '\t'.join(formatted_new_line)
        # print(kern.body[i])
    return kern

def get_xml_from_target(target,
                        labels=LabelsMultiple(extended=True)):
    keys = [m[0] for m in target]
    time_sigs = [m[1] for m in target]
    def get_part_from_sequence(seq, staff='lower'):
        kern_data = ['**kern']
        for measure in seq:
            kern_data.append(''.join(labels.decode(measure)))
        kern_data = '\n=\n'.join(kern_data) + '\n='
        kern_data = add_split_token(kern_data.split('\n'))
        kern_data = '\n'.join(kern_data)
        kern = Kern(data=kern_data + '\n*-\n')
        kern = eliminate_duplicate_chords(kern)
        if not os.path.exists('temp'):
            os.makedirs('temp')
        kern.save(Path(f'temp/{staff}.krn'))
        process = subprocess.run(['tiefix', f'temp/{staff}.krn'],
                                  capture_output=True,
                                  encoding='iso-8859-1')
        kern = Kern(data=process.stdout)
        kern.save(Path(f'temp/{staff}.krn'))
        os.system(f'hum2xml temp/{staff}.krn >temp/{staff}.xml')
        score = m21.converter.parse(f'temp/{staff}.xml')
        score[1].partName = 'Piano'
        score[1][0].instrumentName = 'Piano'
        score[1][0].instrumentAbbreviation = 'Pno.'
        return score[1]
    lower_part = get_part_from_sequence([measure[2] for measure in target], 'lower')
    upper_part = get_part_from_sequence([measure[3] for measure in target], 'upper')
    score = m21.stream.Score()
    treble_clef = m21.clef.TrebleClef()
    bass_clef = m21.clef.BassClef()
    upper_part.insert(0, treble_clef)
    lower_part.insert(0, bass_clef)
    score.append(upper_part)
    score.append(lower_part)
    
    # Add key and time signature
    current_key, current_time_sig = None, None
    for i in range(5):
        if keys[i] != current_key:
            current_key = int(keys[i])
            key_sig = m21.key.KeySignature(current_key)
            score.parts[0].getElementsByClass('Measure')[i].keySignature = key_sig
            score.parts[1].getElementsByClass('Measure')[i].keySignature = key_sig
        if time_sigs[i] != current_time_sig:
            current_time_sig = time_sigs[i]
            time_sig = m21.meter.TimeSignature(current_time_sig)
            score.parts[0].getElementsByClass('Measure')[i].timeSignature = time_sig
            score.parts[1].getElementsByClass('Measure')[i].timeSignature = time_sig
    return score