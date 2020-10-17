from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time
from lossT import sparse_categorical_crossentropy


#--------------------------------------------------
# Parameters and dictionaries

# Number of bins and smoothen length
num_bins=4
sm_length=50
threshold=100

# x-values of the metastable states in the 4-state model potential.
X=[2.0, 0.5, -0.5, -2.0] 

# Labels of all possible states in the ranges we considered.
# For 2d systems, this is not the same as the number of representative values.
all_combs = [i for i in range(num_bins)]
vocab=sorted(all_combs)
vocab_size = len(vocab)

char2idx = {u:i for i, u in enumerate(vocab)} # Mapping from characters to indices
idx2char = np.array(vocab)

# Sequence length and shift in step between past (input) & future (output)
seq_length = 100
shift=1

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset.
BUFFER_SIZE = 50000

# Model parameters
embedding_dim = 8
rnn_units = 64

# Training epochs
EPOCHS=2

# Prediction
num_generate = 120000
# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
temperature = 1.0


#--------------------------------------------------
# Functions
def running_mean(x, N):
    """
    Convolution as running average. Smoothen data.
    """
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def find_nearest(key_arr, target):
    """
    key_arr: array-like, storing keys.
    target: the representative value which we want to be closest to.
    """
    idx=np.abs(key_arr-target).argmin()
    return idx

def Rm_peaks_steps(traj, threshold):
    """
    Remove sudden changes in the trajectory such as peaks and small steps.
    Here the gradient is used to identify the changes. If two nonzero
    gradients are too close (< threshold), we treat it as noise.
    """
    traj=np.array(traj)
    grad_traj=np.gradient(traj) # gradient of trajectory
    idx_grad=np.where(grad_traj!=0)[0] # the index of nonzero gradient.
    idx0=idx_grad[0]
    for idx in idx_grad:
        window=idx-idx0
        if window <= 1: # neighbor
            continue
        elif window > 1 and window <= threshold:
            traj[idx0:idx0+window//2+1]=traj[idx0]
            traj[idx0+window//2+1:idx+1]=traj[idx+1]
            idx0=idx
        elif window > threshold:
            idx0=idx
    return traj

def split_input_target(chunk):
    """
    split sequences into input and target.
    """
    input_text = chunk[:-shift]
    target_text = chunk[shift:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Language_model = Embedding_layer + LSTM + Linear_output.
    """
    model=tf.keras.Sequential([
            
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
    tf.keras.layers.Dense(vocab_size)
    
    ])
    
    return model

def loss(labels, logits):
    """
    Use sparse categorical cross entropy as loss function.
    """
    return sparse_categorical_crossentropy(labels, logits, from_logits=True)

def generate_text(pmodel, num_generate, temperature, start_string):
    """
    # Define function for generating prediction.
    """

    # Converting the start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store the results
    text_generated = np.empty(1)

    # Here batch size = 1
    pmodel.reset_states()
    for i in range(num_generate):
        
        predictions = pmodel(input_eval)
        
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        
        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        
        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        
        text_generated = np.vstack((text_generated, idx2char[predicted_id].tolist()))
        
    return text_generated



#--------------------------------------------------
# Read trajectory
infile = 'xvyw1beta9.5gammax1.0gammay1.0epsln1.0sgma1.0A1.0x01.122w0.8B0.15a1.0_h0.01_mix1.txt'
input_x, input_y = np.loadtxt(infile, unpack=True, usecols=(0,1), skiprows=1)

input_x = running_mean(input_x, sm_length)
idx_x = map(lambda x: find_nearest(X, x), input_x) # clustering

idx_2d=list(idx_x)
idx_2d = Rm_peaks_steps(idx_2d, threshold)

text = idx_2d

text_as_int = np.array([char2idx[c] for c in text])

# Create training examples/targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+shift, drop_remainder=True)
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#--------------------------------------------------
# Read and use the same trajectory as the validation data:
infile_v = 'xvyw1beta9.5gammax1.0gammay1.0epsln1.0sgma1.0A1.0x01.122w0.8B0.15a1.0_h0.01_mix1.txt'
input_xv, input_yv = np.loadtxt(infile_v, unpack=True, usecols=(0,1), skiprows=1)

input_xv = running_mean(input_xv, sm_length)
idx_xv = map(lambda x: find_nearest(X, x), input_xv) # clustering

idx_2dv=list(idx_xv)
idx_2dv = Rm_peaks_steps(idx_2dv, threshold)

vali = idx_2dv[:40000]
vali_as_int = np.array([char2idx[c] for c in vali])

# Create validation examples/targets
vali_dataset = tf.data.Dataset.from_tensor_slices(vali_as_int)

sequences = vali_dataset.batch(seq_length+1, drop_remainder=True)
vdataset = sequences.map(split_input_target)
vdataset = vdataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#--------------------------------------------------
# Read the same trajectory and use the first few to activate the model for prediction
infile_p = 'xvyw1beta9.5gammax1.0gammay1.0epsln1.0sgma1.0A1.0x01.122w0.8B0.15a1.0_h0.01_mix1.txt'
input_xp, input_yp = np.loadtxt(infile_p, unpack=True, usecols=(0,1), skiprows=1)

idx_xp = map(lambda x: find_nearest(X, x), input_xp)

idx_2dp=list(idx_xp) # list(zip(idx_xp, idx_yp))
idx_2dp = Rm_peaks_steps(idx_2dp, threshold)
text4activation = idx_2dp[:1000]#idx_2dp[:100000]



#--------------------------------------------------
# Decide whether to use GPU and build model of training
if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNLSTM
else:
    import functools
    rnn = functools.partial(
    tf.keras.layers.LSTM, recurrent_activation='sigmoid')

model = build_model(vocab_size = len(vocab), embedding_dim=embedding_dim,
        rnn_units=rnn_units, batch_size=BATCH_SIZE)

print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

#--------------------------------------------------
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

#--------------------------------------------------
# Training:
examples_per_epoch = len(text)//(seq_length+shift)
steps_per_epoch = examples_per_epoch//BATCH_SIZE

v_examples_per_epoch=len(vali_as_int)//(seq_length+shift)
v_steps_per_epoch=v_examples_per_epoch//BATCH_SIZE

history = model.fit(dataset.repeat(EPOCHS), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=vdataset.repeat(EPOCHS), validation_steps=v_steps_per_epoch, callbacks=[checkpoint_callback])

#--------------------------------------------------
# Rebuild model with batch_size=1:
pmodel = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
pmodel.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
pmodel.build(tf.TensorShape([1, None]))
print(pmodel.summary())

# Print the length of seed for activating the model
print('length of seed: {}'.format(len(text4activation)))

#--------------------------------------------------
# Generate prediction sequentially:
start0 = time.time()
prediction=generate_text(pmodel, num_generate, temperature, start_string=text4activation)
print ('Time taken for total {} sec\n'.format(time.time() - start0))

#--------------------------------------------------
# Save prediction:
#np.savetxt('prediction',prediction[1:])

print("Done")
