# Investigations in ASR w/ Deep Learning
This repo is a documentation of my journey into Deep Learning via Automatic Speech Recognition.

## First Steps
I'm starting by looking into [Tensorflow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) from 2018 on Kaggle.  I'm interested in this dataset partially because it is small enough to work with on a modest budget, and because I want to understand how much (or little) data is needed to extend this dataset to new words.

We do have DeepSpeech2+ and commercial-grade systems (Siri, Alexa) which use English in its massive entirety, but have very high compute requirements.  But, I'm interested in a medium-sized vocabulary that may be able to train more quickly and run locally.

## The Big Picture
I would like to one day build a system that has the following properties:

1) Performs Speech To Text 100% on device w/ no network call
2) Has a dynamic vocabulary
    - at first this can mean "easy for a developer to extend w/ new words"
    - but ideally this means the user can add words themselves (like we do with our pets)


**Today's systems**
spoken Audio --> well formed English text --> computer instructions
- massive compute requirements for inference
- long, expensive training time
- require network call to decode speech

**My ideal**
spoken audio --> computer instructions
- 100% on device
- dynamic, flexible
- low-cost training, inference