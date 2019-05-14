---
layout: page
title: DeepSymphony - Conditional Polyphonic Music Generation
---

## Aim
The aim of this project is to produce a monophonic (single-instrument) chord accompaniment music track for a given single-channel melody.

**Sample Input**
{% include audio_player.html filename="7_melody.mp3" %}

**Generated Output**
{% include audio_player.html filename="7_all.mp3" %}

<br>

## Data-set
We have used the [Christian Hymn Dataset](https://www.hymnal.net/en/home) which contains 3385 hymns represented in MIDI files. For each hymn, we had two MIDI files : the leading melody file and the final accompaniment track.

The dataset was found from here : [https://github.com/wayne391/Symbolic-Musical-Datasets](https://github.com/wayne391/Symbolic-Musical-Datasets)
<br>

## Data-preparation

The downloaded MIDI files for the same hymn were often of different lengths. This means that the melody and accompaniment tracks for the same hymn were sometimes padded with silence. These padding were removed, the tracks were converted to numpy array.
Finally the 2 tracks for all hymns were made of the same length with trailing zero padding.

**Usefull Libraries for Music Pre-processing** 

[Pypianoroll](https://salu133445.github.io/pypianoroll/) is a python package for handling multi-track piano-rolls built by the creators of MuseGAN. Features : 
-   handle piano-rolls of multiple tracks with metadata
-   utilities for manipulating piano-rolls
-   save to and load from .npz files using efficient sparse matrix format
-   parse from and write to MIDI files 

## Simple LSTM

A simple 1-layer LSTM network was used. The network trained very slowly, taking 6 hours for 10 epochs. 

 - Loss function : Binary Cross-entropy
 - Optimizer : Adam. Learning Rate = 0.001
 
 

**Given Melody :**

 ![Given Melody](https://akhileshdevrari.github.io/CS-671/img/given.png)
 

**Expected Accompaniment :** 

![Expected](https://akhileshdevrari.github.io/CS-671/img/expected.png)


**Results :** 

![Output](https://akhileshdevrari.github.io/CS-671/img/output.png)
