import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

with open("data/train_set.pickle", "rb") as infile:
    train_data_set = pickle.load(infile)
with open("data/dev_set.pickle", "rb") as infile:
    dev_data_set = pickle.load(infile)
with open("data/test_set.pickle", "rb") as infile:
    test_data_set = pickle.load(infile)

x_train = tf.data.Dataset.from_tensor_slices(tf.cast([x[0:2] for x in train_data_set], tf.float32))
y_train = tf.data.Dataset.from_tensor_slices(tf.cast([y[2] for y in train_data_set], tf.float32))

x_dev = tf.data.Dataset.from_tensor_slices(tf.cast([x[0:2] for x in dev_data_set], tf.float32))
y_dev = tf.data.Dataset.from_tensor_slices(tf.cast([y[2] for y in dev_data_set], tf.float32))

x_test = tf.data.Dataset.from_tensor_slices(tf.cast([x[0:2] for x in test_data_set], tf.float32))
y_test = tf.data.Dataset.from_tensor_slices(tf.cast([y[2] for y in test_data_set], tf.float32))


def initialize_parameters():
    initializer = tf.keras.initializers.GlorotNormal() 

    W1 = tf.Variable(initializer(shape=(10, 2)))
    b1 = tf.Variable(initializer(shape=(10, 1)))
    W2 = tf.Variable(initializer(shape=(5, 10)))
    b2 = tf.Variable(initializer(shape=(5, 1)))
    W3 = tf.Variable(initializer(shape=(1, 5)))
    b3 = tf.Variable(initializer(shape=(1, 1)))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.math.add(tf.linalg.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)

    return Z3

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 10, minibatch_size = 1, print_cost = True):
    costs = []
    train_acc = []
    test_acc = []

    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))
    
    m = dataset.cardinality().numpy()
    
    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)

    for epoch in range(num_epochs):

        epoch_cost = 0.
        
        train_accuracy.reset_states()
        
        for (minibatch_X, minibatch_Y) in minibatches:
            
            with tf.GradientTape() as tape:
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)

                minibatch_cost = compute_cost(Z3, minibatch_Y)

            train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost
        
        epoch_cost /= m

        if print_cost == True and epoch % 1 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("Train accuracy:", train_accuracy.result())
            
            for (minibatch_X, minibatch_Y) in test_minibatches:
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                test_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            print("Test_accuracy:", test_accuracy.result())

            costs.append(epoch_cost)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()

    return parameters, costs, train_acc, test_acc

def run_model():
    parameters, costs, train_acc, test_acc = model(x_train, y_train, x_test, y_test, learning_rate=0.1, num_epochs=10)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))
    plt.show()

run_model()