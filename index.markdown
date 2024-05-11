---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
date: 2024-05-10
title: End-to-End Real-World Polyphonic Piano Audio-to-Score Transcription with Hierarchical Decoding
---

# Abstract

Piano audio-to-score transcription (A2S) is an important yet underexplored task with extensive applications for music composition, practice, and analysis. However, existing end-to-end piano A2S systems faced difficulties in retrieving bar-level information such as key and time signatures, and have been trained and evaluated with only synthetic data. To address these limitations, we propose a sequence-to-sequence (Seq2Seq) model with a hierarchical decoder that aligns with the hierarchical structure of musical scores, enabling the transcription of score information at both the bar and note levels by multi-task learning. To bridge the gap between synthetic data and recordings of human performance, we propose a two-stage training scheme, which involves pre-training the model using an expressive performance rendering (EPR) system on synthetic audio, followed by fine-tuning the model using recordings of human performance. To preserve the voicing structure for score reconstruction, we propose a pre-processing method for **Kern scores in scenarios with an unconstrained number of voices. Experimental results support the effectiveness of our proposed approaches, in terms of both transcription performance on synthetic audio data in comparison to the current state-of-the-art, and the first experiment on human recordings.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="static/img/model_architecture.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">The proposed transcription model with a hierarchical decoder.</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="static/img/training_scheme.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">The proposed two-stage training scheme.</div>
</center>

# Demos

## Synthesized Audio Samples Using Different Soundfonts and Composers

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="static/img/audio_sample/score.svg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Excerpt from Haydn: Piano Sonata in C, H.XVI No.50 - 1. Allegro.</div>
</center>

<table>
    <tr>
        <th>Soundfonts\Composer</th>
        <th>Score</th>
        <th>Bach</th>
        <th>Mozart</th>
        <th>Chopin</th>
    </tr>
    <tr>
        <th>FluidR3_GM</th>
        <th><audio controls><source src="static/audio/audio_sample/score~FluidR3_GM.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Bach~FluidR3_GM.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Mozart~FluidR3_GM.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Chopin~FluidR3_GM.wav" type="audio/wav"></audio></th>
    </tr>
    <tr>
        <th>SalamanderGrandPiano-V3</th>
        <th><audio controls><source src="static/audio/audio_sample/score~SalamanderGrandPiano-V3+20200602.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Bach~SalamanderGrandPiano-V3+20200602.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Mozart~SalamanderGrandPiano-V3+20200602.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Chopin~SalamanderGrandPiano-V3+20200602.wav" type="audio/wav"></audio></th>
    </tr>
    <tr>
        <th>TimGM6mb</th>
        <th><audio controls><source src="static/audio/audio_sample/score~TimGM6mb.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Bach~TimGM6mb.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Mozart~TimGM6mb.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Chopin~TimGM6mb.wav" type="audio/wav"></audio></th>
    </tr>
    <tr>
        <th>UprightPianoKW</th>
        <th><audio controls><source src="static/audio/audio_sample/score~UprightPianoKW-20220221.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Bach~UprightPianoKW-20220221.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Mozart~UprightPianoKW-20220221.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Chopin~UprightPianoKW-20220221.wav" type="audio/wav"></audio></th>
    </tr>
    <tr>
        <th>YDP-GrandPiano</th>
        <th><audio controls><source src="static/audio/audio_sample/score~YDP-GrandPiano-20160804.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Bach~YDP-GrandPiano-20160804.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Mozart~YDP-GrandPiano-20160804.wav" type="audio/wav"></audio></th>
        <th><audio controls><source src="static/audio/audio_sample/Chopin~YDP-GrandPiano-20160804.wav" type="audio/wav"></audio></th>
    </tr>
</table>

<table>
    <tr>
        <th>Human Performance (KARYAG06M)</th>
        <th><audio controls><source src="static/audio/audio_sample/human.wav" type="audio/wav"></audio></th>
    </tr>
<table>