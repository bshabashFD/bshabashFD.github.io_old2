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

We'll begin again by importing all required packages, and fixing the random seeds

```python
import tensorflow as tf

import numpy as np
from collections import deque

from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import seed
seed(1)


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
```

```python
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print('length of text: {} chars'.format(len(text)))
print(text[:200])
```
```
length of text: 1115394 chars
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you
```

## Phase 2: Split the text into individual words
We're going to split our text into individual words this time, and let the network build text one word at a time.

Alright, let's split into individual words:

```python
words = text.split(' ')
print(words[:10])
```
```
['First', 'Citizen:\nBefore', 'we', 'proceed', 'any', 'further,', 'hear', 'me', 'speak.\n\nAll:\nSpeak,', 'speak.\n\nFirst']
```
**Ah ha!** we see our first problem... Some words contain a newline `\n` character (some even contain 2). This is an issue, we want to split words separated both by newlines and spaces. Let's fix that shall we?

```python
text_with_no_newlines = text.replace('\n\n', '\n').replace('\n', ' ')
print(text_with_no_newlines[:400])
```
```
First Citizen: Before we proceed any further, hear me speak. All: Speak, speak. First Citizen: You are all resolved rather to die than to famish? All: Resolved. resolved. First Citizen: First, you know Caius Marcius is chief enemy to the people. All: We know't, we know't. First Citizen: Let us kill him, and we'll have corn at our own price. Is't a verdict? All: No more talking on't; let it be done
```

Notice we replaced our double newlines with a single newlines, and then replaced all single newlines with a space. Now let's see what happens when we split.

```python
words = text_with_no_newlines.split(' ')
print(words[:10])
```
```
['First', 'Citizen:', 'Before', 'we', 'proceed', 'any', 'further,', 'hear', 'me', 'speak.']
```

Hmmmm... this is better, but notice some words have punctuation marks such as colons and periods. Let's see if we can find all punctuation marks in our text.

```python
all_chars = list(set(text))
for char in all_chars:
    if not (char.isalpha()):
        print(repr(char))
```
```
'&'
'$'
"'"
'.'
':'
' '
';'
','
'\n'
'!'
'3'
'-'
'?'
```
So we have an interesting collection of punctuation marks, plus a few other marks such as apostrophes, dollar sign (how did that get there?), and the number `3`.
Let's define all marks we want to separate from the word that precedes them.

```python
marks_to_seperate = [';','-','$',':',',','.','?','!','&', '\n'] # Notice we did not put the apostrophe in this list since 
                                                                # it is usually part of a word (e.g. isn't).
```

```python
for mark in marks_to_seperate:
    text = text.replace(mark, ' '+mark+' ').replace('  ', ' ')
    print("replaced "+mark)
    
text = text.replace('\n \n', '\n\n')
```
```
replaced ;
replaced -
replaced $
replaced :
replaced ,
replaced .
replaced ?
replaced !
replaced &
replaced 
```
```python
print(repr(text[: 400]))
```
```
"First Citizen : \n Before we proceed any further , hear me speak . \n\n All : \n Speak , speak . \n\n First Citizen : \n You are all resolved rather to die than to famish ? \n\n All : \n Resolved . resolved . \n\n First Citizen : \n First , you know Caius Marcius is chief enemy to the people . \n\n All : \n We know't , we know't . \n\n First Citizen : \n Let us kill him , and we'll have corn at our own price . \n Is'"
```
Great! much better. Notice how we still have the newlines in there. Why? because by padding each of the punctuation marks with spaces, we in essence define them as words we will use in our RNN. So we kept the newline there as well so that network knows to put in newlines in our output.

```python
words = text.split(' ')
print(words[:20])
vocab = list(set(words))
```

## Phase 3: Define our problem

Last time, we tried to build words one character at a time, so given `goodby` we wanted our network to output `e` to get `goodbye`. This time we will build our text one word at a time, so given the string `Humpty Dumpty sat on the` we'd like to see the output `wall` (to get `Humpty Dumpty sat on the wall`). We'll use a sequence of length 10 this time.

```python
seq_length = 10
```

## Phase 4: Pre-process our data

RNNs, and really all machine learning methods, can only accept numbers as inputs, so we need first to number’ify our text. What this means is that we will assign every character to a number, and also have a way of converting those numbers back to characters

```python
# dictionary comprehension, assign every character to a number from 0-64
word2idx = {u:i for i, u in enumerate(vocab)}
idx2word = np.array(vocab)

# let’s check our text after it’s been transformed into numbers
text_as_int = np.array([word2idx[word] for word in words])
print(text_as_int)
```
```
[13498 10369  7245 ...  6460  3676     0]
```

Now we have a long array with numbers in it, we want to create a dataset which has a 10 number long sequence as the input (X), and a single number as the output (Y). Again, we're going to use a python `deque` for that, which is like a list but has limited capacity, so that when it fills up, and we enter another element, the oldest element is kicked out.

```python
X_data = []
y_data = []


data_deque = deque([],maxlen=seq_length)


for i, word_id in enumerate(text_as_int[:-1]):
    data_deque.append(word_id)
    
    if (len(data_deque) == seq_length):
        X_data.append(list(data_deque))
        y_data.append(text_as_int[i+1])
    
    if ((i % 100) == 0):
        print(i, end="\r")

print(i)

X_data_np = np.array(X_data)
y_data_np = np.array(y_data)

print(X_data_np.shape)
print(y_data_np.shape)
```

```
285077
(285069, 10)
(285069,)
```

Let's take a look at out created dataset:

```python
# Let's take a look at the X and Y data
for i in range(5):
    print("X:\t",repr(' '.join(idx2word[X_data_np[i]])))
    print("y:\t",repr(idx2word[y_data_np[i]]))
    print('-------------')
```

```
X:	 'First Citizen : \n Before we proceed any further ,'
y:	 'hear'
-------------
X:	 'Citizen : \n Before we proceed any further , hear'
y:	 'me'
-------------
X:	 ': \n Before we proceed any further , hear me'
y:	 'speak'
-------------
X:	 '\n Before we proceed any further , hear me speak'
y:	 '.'
-------------
X:	 'Before we proceed any further , hear me speak .'
y:	 '\n\n'
-------------
```
Looks good, the last thing we're going to do, is shuffle the data around. We're going to shuffle the data again later on, so let's make a function to do that for us

```python
def shuffle_data(X_data, y_data):
    
    y_data = y_data.reshape((y_data.shape[0], 1))
    combined_data = np.hstack((X_data, y_data))
    
    np.random.shuffle(combined_data)

    X_data = combined_data[:, :-1]
    y_data = combined_data[:, -1]
    
    return X_data, y_data


X_data_np, y_data_np = shuffle_data(X_data_np, y_data_np)
print(X_data_np.shape)
print(y_data_np.shape)
```
```
(285069, 10)
(285069,)
```

## Phase 5: Build and Compile our Model

Finally we can create our model and train it. We're going to use Long-Short Term Memory units (LSTM units) as our recurrent units, but you can experiment with Gated Recurrent Units (GRUs) as well.

But before we can build our model, we need to consider one more thing. Recall we just assigned a number to each of our words. This was an easy way to turn those words into numbers, but we ended up imposing an artificial order on them. We could figure out some representation which is more meaningful, but we can also have Keras do it for us by introducing an Embedding Unit. In essence, an embedding unit of $n$ dimensions, takes all our vocabulary and learns a unique vector representation for each word in this $n$-dimensional space. Previously we used 25-dimensional space for characters, but we should pick a bigger one for words. Let's try 300 (totally scientific magic number).

Let's build our Recurrent Neural Network:

```python
model = Sequential()

model.add(Embedding(len(vocab), 300, input_length=seq_length))
model.add(LSTM(2048, return_sequences=True))
model.add(LSTM(2048))
model.add(Dense(len(vocab), activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam')

model.summary()
```

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 10, 300)           4370100   
_________________________________________________________________
lstm_2 (LSTM)                (None, 10, 2048)          19243008  
_________________________________________________________________
lstm_3 (LSTM)                (None, 2048)              33562624  
_________________________________________________________________
dense_1 (Dense)              (None, 14567)             29847783  
=================================================================
Total params: 87,023,515
Trainable params: 87,023,515
Non-trainable params: 0
```

**WOW!** 87M parameters to optimize, this will take a while even on Google Colab...

Let's review what we have there:

First, an embedding input layer, it takes a 10 word long sequence and converts each word into a 300-dimensional vector. Thus the output of this layer is a 10$\times$300 two dimensional matrix.

Then, we have a 2048 neuron LSTM layer, it takes the 10$\times$300 matrix and produces a 100$\times$2048 set of values as output. Each LSTM neuron reads the 10$\times$300 matrix, learns to make sense of the sequence of 10 300-dimensional vectors, and outputs a value at each step in the sequence, creating a new sequence of 10 items (`return_sequences=True`). Thus we end up with a new sequence of 10$\times$2048 from the 2048 neurons.

Next is another 2048 neuron LSTM neuron layer. However, it takes the 10$\times$2048 sequence and produces a single set of 2048 values. Each neuron reads the seqeunce, but instead of outputting the sequence, it outputs a single value after having read the entire sequence.

The final layer is a simple layer of 14,567 neurons, who each take the 2048-long output of the 3rd layer, and outputs a number from 0.0-1.0. Each of the 14,567 final neurons correspond to one of the 14,567 words, so each $i^{th}$ neuron outputs the probability the output word for the input sequence is word $i$ (i.e. neuron 0 outputs the probability the next word in the sequence is word 0, neuron 1 outputs the probability the next word in the sequence is word 1, etc').

Once the model is declared, we compile it to bring it to life, and we output the summary to get an idea of our model's architecture.

## Phase 6: Training and Testing

We will train our RNN for 20 epochs, which means it will look at the data 20 times, but to save our poor RAM memory from exploding, we will do so in batches of 480. This means we will divide out dataset into chunks of 480 data points, and show each chunk to the model. Once all chunks have been shown, we will call it an epoch. We’ll then repeat the process 19 more times.

However, after each epoch, we'd like to see what the model has learned. Let's train the model for one epoch and then see how we can test its performance.

```python
BATCH_SIZE = 480   # at each epoch, it will split the data into units of 480 samples, and train on those
i = 1
print("EPOCH: ",i)
X_data_np, y_data_np = shuffle_data(X_data_np, y_data_np)

model.fit(X_data_np, y_data_np,
        batch_size=BATCH_SIZE,
        epochs=1)
```
```
EPOCH:  1
285069/285069 [==============================] - 659s 2ms/sample - loss: 5.7597
<tensorflow.python.keras.callbacks.History at 0x7f7922f786a0>
```
Now let's suppose I want to give my model the input `"ROMEO"` and see what it outputs. This is fine, because `"ROMEO"` is a word in our vocabulary, and we can pad it with newlines to make the sequence size 10 so the input length fits.

But what if we wanted to give the model some sentence to start, and some words in the sentence do not match the words we have in the vocabulary? This was never the case in the previous implementation because we split the text into characters, meaning as long as we used English characters, we were fine. However, in this case we'll have to get creative.

One potential solution is to use a better, pre-made, word embedding which includes many more words. Another solution is to look at each word in our starting sequence, and find the most similar word to it in the available vocabulary. We will use this approach, and employ a [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) metric to compare strings. I found a good implementation right [here](https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/)

```python
def levenshtein(seq1, seq2):
    '''Levenshtein Distance calculation between seq1 and seq2'''
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])
```
```python
def turn_string_into_list(the_string):
    '''
    Takes a string and converts it into a list of words, treating
    the given set of marks_to_seperate as special words.

    e.g.
    a = "Hello, my name is Boris"
    turn_string_into_list(a) -> ['Hello', ',', 'my', 'name', 'is', 'Boris']

    '''
    the_string = the_string.replace('\n\n', 'doublenewline')
    marks_to_seperate = [';','-','$',':',',','.','?','!','&', '\n', 'doublenewline'] # Notice we did not put the apostrophe in this list since 
                                                                # it is usually part of a word (e.g. isn't).
    for mark in marks_to_seperate:
    the_string = the_string.replace(mark, ' '+mark).replace('  ', ' ')

    the_string = the_string.replace('doublenewline', '\n\n')
    string_as_list = the_string.split(' ')

    return string_as_list
```
```python
def padd_string_to_length(the_string, the_length):
    '''
    Takes a string and returns a list of words with length the_length
    with newline characters (\n) padding in the front if the string
    is too short, or only a subset of the string if the string is
    too long

    e.g.
    a = "Hello my name is Boris"
    padd_string_to_length(a, 10) -> ['\n', '\n', '\n', '\n', '\n', 'Hello', 'my', 'name', 'is', 'Boris']
    padd_string_to_length(a, 5) -> ['Hello', 'my', 'name', 'is', 'Boris']
    padd_string_to_length(a, 3) -> ['name', 'is', 'Boris']
    '''


    string_as_list = turn_string_into_list(the_string)

    if (len(string_as_list) > the_length):
        string_as_list = string_as_list[-the_length:]
    else:
        while (len(string_as_list) < the_length):
            string_as_list.insert(0, '\n')

    return string_as_list
```
```python
def find_closest_word(word, vocab):
    '''
    Uses the levenshtein_distance function to find the closest word
    to word in vocab. The matching is case sensitive.

    e.g.
    a = "ROME"
    my_vocab = ["ROMEO", "RO", "JULIET", "rome"]
    find_closest_word(a, my_vocab) -> "ROMEO"
    '''


    levenshtein_distance = float('inf')
    return_word = '\n'

    for vocab_word in vocab:
        new_levenshtein_distance = levenshtein(word, vocab_word)
    if (new_levenshtein_distance < levenshtein_distance):
        levenshtein_distance = new_levenshtein_distance
        return_word = vocab_word

    return return_word                              
```
```python
def convert_string_into_model_approved_words(the_string, seq_length):
    '''
    Takes the string the_string and returns the string as a list
    of length seq_length with newline padding
    at the begining and being composed only of words contained
    in vocab (global_variable)

    e.g.
    convert_string_into_model_approved_words("BORIS", 5) -> ['\n', '\n', '\n', '\n', 'PARIS']
    convert_string_into_model_approved_words("BORIS hello", 5) -> ['\n', '\n', '\n', 'PARIS', 'hell']
    convert_string_into_model_approved_words("cold winter", 5) -> ['\n', '\n', '\n', 'cold', 'winter']

    '''

    string_as_list = padd_string_to_length(the_string, seq_length)

    for index, word in enumerate(string_as_list):
        if (word not in vocab):
            replacement_word = find_closest_word(word, vocab)
            string_as_list[index] = replacement_word

    return string_as_list
```

Now that we have the functions above defined, we can use them to create a prediction loop. We will request an output of size 20 (20 new words being added to our provided string)

```python
import copy

def get_text_from_model(model, test_input, output_size=20):
    
    test_output = copy.copy(test_input)
    

    for i in range(output_size):
        string_list = convert_string_into_model_approved_words(test_output, seq_length)
        
        text_as_int = np.array([word2idx[word] for word in string_list])
        text_as_int = text_as_int.reshape((1, seq_length))

        y_predict_proba = model.predict(text_as_int)[0]
        y_id = np.random.choice(list(range(len(vocab))), size=1, p=y_predict_proba)[0]

        y_word = idx2word[y_id]
        test_output = test_output+" "+y_word
        print("added word #"+str(i), end="\r")

        
    # Final touch, for every punctuation mark, we should remove its
    # preceeding space
    
    marks_to_seperate = [';','-','$',':',',','.','?','!','&', '\n'] # Notice we did not put the apostrophe in this list since 
                                                                # it is usually part of a word (e.g. isn't).
    for mark in marks_to_seperate:
        test_output = test_output.replace(' '+mark, mark)
      
    test_output = test_output.replace('\n ', '\n')
    
    return test_output
```

```python
output_text = get_text_from_model(model, "BORIS:", output_size=20)
print(output_text)
```

```
BORIS: my behind.

GLOUCESTER:
An proved for leet, are yours coffins falsehood?
'Tis foe
```

This function basically takes our model, a seed input, and a length, and produces an output of that length (excluding our initial seed).

Let's train our model!

```python
EPOCHS = 20       # NNs operate in epochs, meaning this is how many times the neural network will go 
                 # through the entire data
BATCH_SIZE = 480   # at each epoch, it will split the data into units of 480 samples, and train on those


for i in range(2, EPOCHS+1):
    print("EPOCH: ",i)
    X_data_np, y_data_np = shuffle_data(X_data_np, y_data_np)
    model.fit(X_data_np, y_data_np,
            batch_size=BATCH_SIZE,
            epochs=1)

    test_input = "ROMEO:\n"
    print(get_text_from_model(model, test_input, output_size=20))
    print("------------")
```

```
EPOCH:  2
285069/285069 [==============================] - 642s 2ms/sample - loss: 4.9097
ROMEO: and your troth is deceit privilege my duke.

KATHARINA:
Will this marriage, look good enemy
------------
EPOCH:  3
285069/285069 [==============================] - 638s 2ms/sample - loss: 4.5822
ROMEO: serve my soul! which it was no;
Thou, if thou Lewis but, a whose cross'd
------------
EPOCH:  4
285069/285069 [==============================] - 636s 2ms/sample - loss: 4.3222
ROMEO: our scope I to these ransack'd
that dogs it can have worthy other's hand;
Lie for she
------------
EPOCH:  5
285069/285069 [==============================] - 634s 2ms/sample - loss: 4.0725
ROMEO: by that stock I was well thus with a will,
Your wonted name did not deign to trumpets
------------
EPOCH:  6
285069/285069 [==============================] - 632s 2ms/sample - loss: 3.7869
ROMEO: a lady's cousin is known
A follow up to all a which can
Intends to fast a day
------------
EPOCH:  7
285069/285069 [==============================] - 630s 2ms/sample - loss: 3.4875
ROMEO: even thou, in my stitchery, love till I jest,
And to our voices cause, or
------------
EPOCH:  8
285069/285069 [==============================] - 628s 2ms/sample - loss: 3.1739
ROMEO: ourselves, at my interior flaunts, Iniquity.

BUCKINGHAM:
'Twas she that same words holds leisure
------------
EPOCH:  9
285069/285069 [==============================] - 627s 2ms/sample - loss: 2.8628
ROMEO: thou hast thy device, your good malice?

RATCLIFF:
Sir Christopher a heavy queen's thing,
------------
EPOCH:  10
285069/285069 [==============================] - 628s 2ms/sample - loss: 2.5636
ROMEO: then I'll tell them.

BIONDELLO:
You say, we beseech you?

ABRAHAM:

------------
EPOCH:  11
285069/285069 [==============================] - 626s 2ms/sample - loss: 2.2738
ROMEO: what's what o'clock thou art to be the day?
Though Edward have worn himself dead before
waiting
------------
EPOCH:  12
285069/285069 [==============================] - 627s 2ms/sample - loss: 1.9850
ROMEO: O, 'tis my mother's brother!

CLARENCE:
He is the way, if there be glad
------------
EPOCH:  13
285069/285069 [==============================] - 632s 2ms/sample - loss: 1.6942
ROMEO: a unlook'd- gentle mother, and so begg'd
as we were in to the wall, who lately
------------
EPOCH:  14
285069/285069 [==============================] - 636s 2ms/sample - loss: 1.4026
ROMEO: I am on both of my Kate, you have.

POLIXENES:
I do not, sir
------------
EPOCH:  15
285069/285069 [==============================] - 639s 2ms/sample - loss: 1.1094
ROMEO: he had his desire to face me at the charge
And with her queen? And do we now
------------
EPOCH:  16
285069/285069 [==============================] - 637s 2ms/sample - loss: 0.8314
ROMEO: the gain of my mind is now: who
was not barren here, I never stand in gold
------------
EPOCH:  17
285069/285069 [==============================] - 645s 2ms/sample - loss: 0.5855
ROMEO: the greater tidings, our Harry is the good.

DUKE VINCENTIO:
Howsoever you shall plainly be
------------
EPOCH:  18
285069/285069 [==============================] - 636s 2ms/sample - loss: 0.3870
ROMEO: the sun would have been his wife; we
protest, so it must be so; but I
------------
EPOCH:  19
285069/285069 [==============================] - 645s 2ms/sample - loss: 0.2406
ROMEO: the most commanding Bolingbroke,
I have deserved my brother.

SICINIUS:
When you do none
------------
EPOCH:  20
285069/285069 [==============================] - 631s 2ms/sample - loss: 0.1447
ROMEO: thy food are broke from the dead of it.

VIRGILIA:
This is most likely.


------------
```

Finally let's see our RNN in action. Write a play about me

```python
test_string = "BORIS THE BLADE:\n"
print(get_text_from_model(model, test_string, output_size=190))
```
```
BORIS THE BLADE:
And beg of mine, which may I complain mine.

LADY ANNE:
Foul a heart- peace! O, nothing now!
If this your face are to be call'd ere one
Of your horse; his answer it was it;
And whither dost thou say King Richard is
A lost house! Let us be ready, sir.

ESCALUS:
Let's hear good help; let's bring you to a whore.

GLOUCESTER:
He hath been good to love the truth of thy kindness
And break into mine eyes so doth thy life.
Come, therefore, go up, I say.

WARWICK:
Upon him, madam, we will stay it now.
If I may live, my lord, he did;
And, as the manner is itself, stands out:
We break the mansion of the cause; but is
the city, by the chair of question, let them
continue a present earth
```

Now this is much nicer. We could probably train the model some more but we'll get diminishing returns here.

I hope this helps in exploring the idea behind text generation (and classification) employing RNNs.
