---
layout: post
title: Employing Recurrent Neural Networks for Text Generation - Part II
mathjax: true
image: writing-logo.png
---

In the [previous post](https://bshabashfd.github.io/2019/08/03/how-to-use-text-with-rnn.html) We've explored how we can use TensorFlow, Keras, and the Recurrent Neural Network (RNN) architecture to produce a system which can generate new Shakespearean prose for us. However, we ran into a problem... The network spends most of its time learning how to compose English words. It learns the form very quickly, but then we need many epochs to make the network spell correct words. In this tutorial we will explore how to solve that problem.


## Why are we having this issue?

In essence, the problem comes from the way we constructed our vocabulary. We split the entire text into individual characters and tried to predict the next character in the sequence. However, we can employ a more reasonable approach and split the data collection into words instead. This will allow the network to choose from already valid words. This seems easy enough, but there are in fact a few hurdles this approach will introduce. Let's work through each one as they come up:

## Phase 1: Import Keras and Obtain the data
