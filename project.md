---
layout: page
title: DeepSymphony - Conditional Polyphonic Music Generation
---

## Aim
The aim of this project is to produce a polyphonic (multi-instrument) multi-channel accompaniment music track given a single-channel melody.

**Sample Input**
{% include audio_player.html filename="1_melody.mp3" %}

**Generated Output**
{% include audio_player.html filename="2_accompaniment.mp3" %}

**Combined Input + Output**
{% include audio_player.html filename="3_mix.mp3" %}
<br>

## Data-set
We have used the [Lakh Midi Dataset](https://colinraffel.com/projects/lmd/) which contains 176,581 deduped MIDI files. From literature review, we have learnt that we need to only select songs that are in the key of C and have a specific beat pattern. This had been done to simplify the process of music genearation.
<br>

## Data-preparation



**Usefull Libraries for Music Pre-processing** 

 1. [Music21 Library](http://web.mit.edu/music21/doc/index.html) + [tutorial](https://www.kaggle.com/wfaria/midi-music-data-extraction-using-music21)
 2. [python-midi](https://github.com/vishnubob/python-midi)
 3. [mido](https://mido.readthedocs.io/en/latest/)
