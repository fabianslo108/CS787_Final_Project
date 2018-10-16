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

N_train = x_np.shape[0]//3*2
N_valid = (x_np.shape[0]-x_np.shape[0]//3*2)//2
N_test = x_np.shape[0]-N_train-N_valid

print("Training data size: ", N_train)
print("Validation data size: ", N_valid)
print("Testing data size: ", N_test)

train_x= x_np[:N_train, :]
train_y = y_np[:N_train]
valid_x = x_np[N_train:N_train+N_valid, :]
valid_y = y_np[N_train:N_train+N_valid]
test_x = x_np[N_train+N_valid:, :]
test_y = y_np[N_train+N_valid:]

in_size = x_np.shape[1] // 4

# Generate tensorflow graph
learning_rate = .001
n_epochs = 9000
batch_size = 40
kp = 0.5

with tf.name_scope("placeholders"):
    x1 = tf.placeholder(tf.float32, (None, in_size))
    x2 = tf.placeholder(tf.float32, (None, in_size))
    x3 = tf.placeholder(tf.float32, (None, in_size))
    x4 = tf.placeholder(tf.float32, (None, in_size))
    y = tf.placeholder(tf.float32, (None,))
    keep_prob = tf.placeholder(tf.float32)
with tf.name_scope("hidden"):
    W1 = tf.Variable(tf.random_normal((in_size, 1)))
    W2 = tf.Variable(tf.random_normal((in_size, 1)))
    W3 = tf.Variable(tf.random_normal((in_size, 1)))
    W4 = tf.Variable(tf.random_normal((in_size, 1)))
    b1 = tf.Variable(tf.random_normal((1,)))
    b2 = tf.Variable(tf.random_normal((1,)))
    b3 = tf.Variable(tf.random_normal((1,)))
    b4 = tf.Variable(tf.random_normal((1,)))
    x_hidden1 = tf.nn.relu(tf.matmul(x1, W1) + b1)
    x_hidden2 = tf.nn.relu(tf.matmul(x2, W2) + b2)
    x_hidden3 = tf.nn.relu(tf.matmul(x3, W3) + b3)
    x_hidden4 = tf.nn.relu(tf.matmul(x4, W4) + b4)
      # Apply dropout
    x_hidden1 = tf.nn.dropout(x_hidden1, keep_prob)
    x_hidden2 = tf.nn.dropout(x_hidden2, keep_prob)
    x_hidden3 = tf.nn.dropout(x_hidden3, keep_prob)
    x_hidden4 = tf.nn.dropout(x_hidden4, keep_prob)
    x_hidden = tf.concat([x_hidden1, x_hidden2, x_hidden3, x_hidden4], axis=1)
with tf.name_scope("prediction"):
    W = tf.Variable(tf.random_normal((4,1)))
    b = tf.Variable(tf.random_normal((1,)))
    y_pred = tf.matmul(x_hidden, W) + b
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
            batch_X1 = batch_X[:, 0:12]
            batch_X2 = batch_X[:, 12:24]
            batch_X3 = batch_X[:, 24:36]
            batch_X4 = batch_X[:, 36:48]
            batch_y = train_y[pos:pos+batch_size]
            feed_dict = {x1: batch_X1, x2: batch_X2, x3: batch_X3, x4: batch_X4, y: batch_y, keep_prob: kp}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
            train_writer.add_summary(summary, step)

            step += 1
            pos += batch_size

    # Make Predictions
    train_y_pred = sess.run(y_pred, feed_dict={x1: train_x[:, 0:12],
                                               x2: train_x[:, 12:24],
                                               x3: train_x[:, 24:36],
                                               x4: train_x[:, 36:48],
                                               keep_prob: 1})
    valid_y_pred = sess.run(y_pred, feed_dict={x1: valid_x[:, 0:12],
                                               x2: valid_x[:, 12:24],
                                               x3: valid_x[:, 24:36],
                                               x4: valid_x[:, 36:48],
                                               keep_prob: 1})
    test_y_pred = sess.run(y_pred, feed_dict={x1: test_x[:, 0:12],
                                              x2: test_x[:, 12:24],
                                              x3: test_x[:, 24:36],
                                              x4: test_x[:, 36:48],
                                              keep_prob: 1})

train_y_pred = np.reshape(train_y_pred, -1)
train_r2 = pearson_r2_score(train_y, train_y_pred)
print("Training Pearson R^2: %f" % train_r2)

valid_y_pred = np.reshape(valid_y_pred, -1)
valid_r2 = pearson_r2_score(valid_y, valid_y_pred)
print("Validation Pearson R^2: %f" % valid_r2)

test_y_pred = np.reshape(test_y_pred, -1)
test_r2 = pearson_r2_score(test_y, test_y_pred)
print("Test Pearson R^2: %f" % test_r2)

test_mse = (test_y - test_y_pred)**2
print("Mean: ", test_y.mean())
print("Test RMSE", np.sqrt(test_mse.mean()))

# Clear figure
# plt.clf()
# plt.xlabel("Y-true")
# plt.ylabel("Y-pred")
# plt.title("Predicted versus true values")
# plt.scatter(test_y, test_y_pred)
# plt.savefig("regression_mlp_pred.png")
# plt.show()

