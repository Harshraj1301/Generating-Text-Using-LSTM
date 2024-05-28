
# Generating Text Using LSTM

This repository contains code and resources for generating text using Long Short-Term Memory (LSTM) neural networks. The project demonstrates how to build and train an LSTM model for text generation, using a sample dataset.

## Repository Structure

```
Generating-Text-Using-LSTM/
│
├── .gitattributes
├── Harshraj_Jadeja_HW3_LSTM_TEXT_GEN.ipynb
└── README.md
```

- `.gitattributes`: Configuration file to ensure consistent handling of files across different operating systems.
- `Harshraj_Jadeja_HW3_LSTM_TEXT_GEN.ipynb`: Jupyter Notebook containing the code for building and training the LSTM model, as well as the text generation process.
- `README.md`: This file. Provides an overview of the project and instructions for getting started.

## Getting Started

To get started with this project, follow the steps below:

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Harshraj1301/Generating-Text-Using-LSTM.git
```

2. Navigate to the project directory:

```bash
cd Generating-Text-Using-LSTM
```

3. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Usage

1. Open the Jupyter Notebook:

```bash
jupyter notebook Harshraj_Jadeja_HW3_LSTM_TEXT_GEN.ipynb
```

2. Follow the instructions in the notebook to run the code cells and generate text using the LSTM model.

### Code Explanation

The notebook `Harshraj_Jadeja_HW3_LSTM_TEXT_GEN.ipynb` includes the following steps:

1. **Data Preprocessing**: Loading and preprocessing the text data to make it suitable for training the LSTM model.
2. **Model Building**: Constructing the LSTM model using Keras.
3. **Model Training**: Training the LSTM model on the preprocessed text data.
4. **Text Generation**: Using the trained model to generate new text sequences.

Here are the contents of the notebook:

# Harshraj Jadeja

# Long Short-term Memory for Text Generation

This notebook uses LSTM neural network to generate text from Nietzsche's writings.

## Dataset

### Get the data
Nietzsche's writing dataset is available online. The following code download the dataset.

### Visualize data

### Clean data

We cut the text in sequences of maxlen characters with a jump size of 3.
The features for each example is a matrix of size maxlen*num of chars.
The label for each example is a vector of size num of chars, which represents the next character.

## The model

### Build the model - fill in this box

we need a recurrent layer with input shape (maxlen, len(chars)) and a dense layer with output size  len(chars)

### Inspect the model

Use the `.summary` method to print a simple description of the model

### Train the model

## Code Cells

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import random
import sys
import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import get_file
```

```python
path = get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
```

```python
print('corpus length:', len(text))
```

```python
print(text[10:513])
```

```python
chars = sorted(list(set(text)))
# total nomber of characters
print('total chars:', len(chars))
```

```python
# create (character, index) and (index, character) dictionary
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
```

```python
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
```

```python
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool_)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool_)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
```

```python
# Define the number of units in the LSTM layer.
# This is a hyperparameter that represents the dimensionality of the output space.
# More units can allow the model to capture more complex patterns but also increases computational complexity.
lstm_units = 128  # Adjust this number based on the complexity of the task and computational constraints.

# Initialize the Sequential model
model = tf.keras.Sequential([
    # Add an LSTM layer as the first layer of the model
    # input_shape is required as the LSTM layer's first layer to let it know the shape of the input it should expect
    # Here, input_shape=(maxlen, len(chars)) means each input sequence will be of length 'maxlen'
    # and each character in the sequence is represented as a one-hot encoded vector of length 'len(chars)'
    tf.keras.layers.LSTM(lstm_units, input_shape=(maxlen, len(chars))),
    
    # Add a Dense output layer
    # The number of units equals the number of unique characters (len(chars))
    # This is because we want to output a probability distribution over all possible characters
    # Softmax activation function is used to output probabilities
    tf.keras.layers.Dense(len(chars), activation='softmax'),
])

# Compile the model
# 'categorical_crossentropy' is used as the loss function since this is a multi-class classification problem
# 'adam' optimizer is chosen for efficient stochastic gradient descent optimization
# Accuracy is monitored as a metric to observe the performance of the model during training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display the model's architecture
model.summary()
```

```python
model.summary()
```

```python
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```

```python
class PrintLoss(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, _):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.5, 1.0]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
```

```python
EPOCHS = 60
BATCH = 128

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(x, y,
                    batch_size = BATCH,
                    epochs = EPOCHS,
                    validation_split = 0.2,
                    verbose = 1,
                    callbacks = [early_stop, PrintLoss()])
```

## Results

The notebook includes the results of the text generation process, showcasing how the trained LSTM model generates sequences of text based on the input data.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project was created as part of an assignment by Harshraj Jadeja.
- Thanks to the open-source community for providing valuable resources and libraries for machine learning.

---

Feel free to modify this `README.md` file as per your specific requirements and project details.
