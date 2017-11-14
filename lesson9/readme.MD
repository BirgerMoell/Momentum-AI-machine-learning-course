# AI in music
![title](https://i.ytimg.com/vi/paBvh-0uw34/maxresdefault.jpg)
AI in music is an interesting and hard topic.

One of the most used libraries for working with sound data is Google Magenta which is a library for working with sound.
https://magenta.tensorflow.org/welcome-to-magenta

## What is Magenta?

Magenta has two goals. First, it’s a research project to advance the state of the art in machine intelligence for music and art generation. Machine learning has already been used extensively to understand content, as in speech recognition or translation. With Magenta, we want to explore the other side—developing algorithms that can learn how to generate art and music, potentially creating compelling and artistic content on their own.

Second, Magenta is an attempt to build a community of artists, coders and machine learning researchers. The core Magenta team will build open-source infrastructure around TensorFlow for making art and music. We’ll start with audio and video support, tools for working with formats like MIDI, and platforms that help artists connect to machine learning models. For example, we want to make it super simple to play music along with a Magenta performance model.

## Example Magenta Projects

## Performance RNN
![title](https://i.imgur.com/nwHQ2Eh.png)

Creating short compositions that sounds okay but lack the longer structure that we are used to have in “real” music pieces
https://magenta.tensorflow.org/performance-rnn

## NSynth: Neural Audio Synthesis
![title](https://i.imgur.com/uEFSWIe.png)

We detail a powerful new WaveNet-style autoencoder model that conditions an autoregressive decoder on temporal codes learned from the raw audio waveform. Second, we introduce NSynth, a large-scale and high-quality dataset of musical notes that is an order of magnitude larger than comparable public datasets. Using NSynth, we demonstrate improved qualitative and quantitative performance of the WaveNet autoencoder over a well-tuned spectral autoencoder baseline. Finally, we show that the model learns a manifold of embeddings that allows for morphing between instruments, meaningfully interpolating in timbre to create new types of sounds that are realistic and expressive.

https://magenta.tensorflow.org/nsynth

## AI Duet: A piano that responds to you.
![title](https://i.imgur.com/OVLGICa.png)

This experiment lets you play a duet with the computer. Just play some notes, and the computer will respond to your melody. You don’t even have to know how to play piano—it’s fun to just press some keys and listen to what comes back. You can click the keyboard, use your computer keys, or even plug in a MIDI keyboard. It's just one example of how machine learning can inspire people to be creative in new ways.

Built by Yotam Mann with friends on the Magenta and Creative Lab teams at Google. It's built with Tensorflow, Tone.js, and open-source tools from the Magenta project.

https://experiments.withgoogle.com/ai/ai-duet/view/

## Repository
https://github.com/googlecreativelab/aiexperiments-ai-duet

# Paper
https://arxiv.org/abs/1704.01279

## WaveNet
![title](https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig1-Anim-160908-r01.gif)
## Abstract
WaveNet is a deep neural network for generating raw audio waveforms. The model is fully probabilistic and autoregressive, with the predictive distribution for each audio sample conditioned on all previous ones; nonetheless we show that it can be efficiently trained on data with tens of thousands of samples per second of audio. When applied to text-to-speech, it yields state-of-the-art performance, with human listeners rating it as significantly more natural sounding than the best parametric and concatenative systems for both English and Mandarin. A single WaveNet can capture the characteristics of many different speakers with equal fidelity, and can switch between them by conditioning on the speaker identity. When trained to model music, we find that it generates novel and often highly realistic musical fragments. We also show that it can be employed as a discriminative model, returning promising results for phoneme recognition.

## Original paper
https://arxiv.org/pdf/1609.03499.pdf


## How to create MIDI files

## Using python

 ```python
pip install MIDIFile
 ```

   ```python

from midiutil.MidiFile import MIDIFile

def createMidi(musicData):
    MyMIDI = MIDIFile(1)
    #Tracks are numbered from zero. Times are measured in beats.
    track = 0
    time = 0
    channel = 0
    time = 0
    volume = 100
    duration = 1
    #Add track name and tempo.
    MyMIDI.addTrackName(track,time,"Sample Track")
    MyMIDI.addTempo(track,time,120)

    musicData = np.reshape(musicData, (1, 16))

    for value in musicData:
        print (musicData.shape)
        print (len(value))
        j = 0
        while j < 16:
                #Now add the note.
                print("the value of j is")
                print(j)
                track = j
                channel = j
                pitch = int(value[j])*10
                time = time + j
                MyMIDI.addNote(track,channel, pitch, time, duration, volume)
                j = j + 1

    #And write it to disk.
    binfile = open("output.mid", 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()

createMidi(data)

   ```


