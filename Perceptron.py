# Necessary libraries are loaded
import pandas as pd
import numpy as np
from pathlib import Path
from pandas import Series
from pandas import DataFrame
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import pearsonr

np.random.seed(456)
tf.set_random_seed(456)


def pearson_r2_score(y, y_pred):
    """
    Computes Pearson R^2 (square of Pearson correlation)
    """
    return pearsonr(y, y_pred)[0]**2


# Load data
data_file = Path(r'C:\Users\Fabian\Dropbox\CS 767\Project\labor-conflict-data.csv')
data = pd.read_csv(filepath_or_buffer=data_file)
data = data.pivot(index='MonthYear', columns='QuadClass', values='f0_')
rng = pd.date_range(start='1/1/1999', periods=data.shape[0], freq='M')     # date index
data.index = rng
data.columns = ['Verbal_Coop', 'Material_Coop', 'Verbal_Conflict', 'Material_Conflict']
# print(data.head())

inputs = []
output = []
x_np = np.array(data)
for i in range(12, len(x_np)):
    record = np.concatenate((x_np[i-12:i, 0], x_np[i-12:i, 1], x_np[i-12:i, 2], x_np[i-12:i, 3]), axis=0)
    inputs.append(record)
    output.append(x_np[i, 3])

x_np = np.array(inputs)
y_np = np.array(output)

# N_train = x_np.shape[0]//3*2
# N_valid = (x_np.shape[0]-x_np.shape[0]//3*2)//2
# N_test = x_np.shape[0]-N_train-N_valid
N_train = x_np.shape[0]

train_x= x_np[:N_train, :]
train_y = y_np[:N_train]
# valid_x = x_np[N_train:N_train+N_valid, :]
# valid_y = y_np[N_train:N_train+N_valid]
# test_x = x_np[N_train+N_valid:, :]
# test_y = y_np[N_train+N_valid:]

in_size = x_np.shape[1]

# Generate tensorflow graph
learning_rate = .002
n_epochs = 100000
batch_size = 48

with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32, (None, in_size))
  y = tf.placeholder(tf.float32, (None,))
with tf.name_scope("weights"):
  W = tf.Variable(tf.random_normal((in_size, 1)))
  b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"):
  y_pred = tf.nn.relu(tf.matmul(x, W) + b)
with tf.name_scope("loss"):
    l = tf.reduce_sum((y - tf.squeeze(y_pred))**2)
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./', tf.get_default_graph())


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(n_epochs):
        pos = 0
        while pos < N_train:
            batch_X = train_x[pos:pos+batch_size]
            batch_y = train_y[pos:pos+batch_size]
            feed_dict = {x: batch_X, y: batch_y}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
            train_writer.add_summary(summary, step)

            step += 1
            pos += batch_size

    # Make Predictions
    train_y_pred = sess.run(y_pred, feed_dict={x: train_x})
    # valid_y_pred = sess.run(y_pred, feed_dict={x: valid_x})

train_y_pred = np.reshape(train_y_pred, -1)
r2 = pearson_r2_score(train_y, train_y_pred)
print("Pearson R^2: %f" % r2)

# Clear figure
plt.clf()
plt.xlabel("Y-true")
plt.ylabel("Y-pred")
plt.title("Predicted versus true values")
plt.scatter(train_y, train_y_pred)
plt.savefig("regression_pred.png")
plt.show()

