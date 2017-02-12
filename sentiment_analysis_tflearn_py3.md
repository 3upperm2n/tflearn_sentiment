

```python
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
```

### imdb dataset


```python
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
```


```python
trainX, trainY = train
testX, testY = test
```

### data preprocessing


```python
# sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
```


```python
len(trainX)
```




    22500




```python
len(testX)
```




    2500




```python
# convert lables to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
```

### buid network


```python
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
```

    WARNING:tensorflow:From /Users/leiming/anaconda2/envs/tflearn/lib/python3.5/site-packages/tflearn/summaries.py:46 in get_summary.: scalar_summary (from tensorflow.python.ops.logging_ops) is deprecated and will be removed after 2016-11-30.
    Instructions for updating:
    Please switch to tf.summary.scalar. Note that tf.summary.scalar uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in. Also, passing a tensor or list of tags to a scalar summary op is no longer supported.
    WARNING:tensorflow:From /Users/leiming/anaconda2/envs/tflearn/lib/python3.5/site-packages/tflearn/summaries.py:46 in get_summary.: scalar_summary (from tensorflow.python.ops.logging_ops) is deprecated and will be removed after 2016-11-30.
    Instructions for updating:
    Please switch to tf.summary.scalar. Note that tf.summary.scalar uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in. Also, passing a tensor or list of tags to a scalar summary op is no longer supported.
    WARNING:tensorflow:From /Users/leiming/anaconda2/envs/tflearn/lib/python3.5/site-packages/tflearn/helpers/trainer.py:766 in create_summaries.: merge_summary (from tensorflow.python.ops.logging_ops) is deprecated and will be removed after 2016-11-30.
    Instructions for updating:
    Please switch to tf.summary.merge.
    WARNING:tensorflow:VARIABLES collection name is deprecated, please use GLOBAL_VARIABLES instead; VARIABLES will be removed after 2017-03-02.
    WARNING:tensorflow:From /Users/leiming/anaconda2/envs/tflearn/lib/python3.5/site-packages/tflearn/helpers/trainer.py:130 in __init__.: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.



```python
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=128)
```

    Training Step: 1760  | total loss: [1m[32m0.14025[0m[0m
    | Adam | epoch: 010 | loss: 0.14025 - acc: 0.9608 | val_loss: 0.52050 - val_acc: 0.8160 -- iter: 22500/22500
    Training Step: 1760  | total loss: [1m[32m0.14025[0m[0m
    | Adam | epoch: 010 | loss: 0.14025 - acc: 0.9608 | val_loss: 0.52050 - val_acc: 0.8160 -- iter: 22500/22500
    --

