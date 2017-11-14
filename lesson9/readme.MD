# AI in music
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

# Paper
https://arxiv.org/abs/1704.01279
