# Learning Molecular Dynamics with Simple Language Model built upon Long Short-Term Memory (LSTM) Neural Network
This is an example showing how to use a simple [LSTM based lagnuage model](https://www.tensorflow.org/tutorials/text/text_generation) to learn a Langevin dynamics of a 4-state model potential introduced in [arxiv.org/abs/2004.12360](arxiv.org/abs/2004.12360). Please read and cite these manuscripts if using this example:
[arxiv.org/abs/2004.12360](arxiv.org/abs/2004.12360)
# LSTM_LM_4s.py
This file contains all settings, data preprocessing, loss function, and the model itself. The program is tested under tensorflow/1.10.1 with keras beckend. Simpy implement this file by
```
python3 LSTM_4s.py
```
in the terminal. The program will save a file with prediction and corresponding training checkpoints of last few epochs.
# lossT.py
This program uses this file as input loss function.
For using other loss, removing line 179 and return in line 182:
```
from lossT import sparse_categorical_crossentropy
def loss(labels, logits):
    return sparse_categorical_crossentropy(labels, logits, from_logits=True)
```

Portions of this page are modifications based on work created and [shared by Google](https://developers.google.com/terms/site-policies) and used according to terms described in the [Creative Commons 4.0 Attribution License](https://creativecommons.org/licenses/by/4.0/).
