import numpy as np
import tensorflow as tf


class autoencoder:
    def __init__(self,
                 code_len,
                 shape=[2],
                 dropout=0.2,
                 l1=1e-5,
                 l2=1e-5):
        """
        Init this simple autoencoder class.

        - input_shape (int): Dimension of the input data
        - code_len (int): Output dimension of the encoder
        - shape (list of ints): Number of hidden layers (list length) and
        number of nodes on each layer (ints)
        - dropout (float): Dropout rate to be applied to each hidden layer
        - l1 (float): L1-regularization to add sparsity into the autoencoder
        - l2 (float): L2-regularization to avoid exploding weights

        Regularization terms only apply to weights of the hidden layers, not
        to the biases.
        """

        self.code_len = code_len
        self.shape = shape
        self.dropout = dropout
        self.l1 = l1
        self.l2 = l2
        return

    def gen_network(self, in_shape):
        """
        Internal. Generate the network graph. Just to make
        the class code more readable
        """

        # Avoid problems re-using names of the variables
        tf.reset_default_graph()

        self.lay_shape = [in_shape] + self.shape + [self.code_len]
        # Weights and biases
        self.w = {}
        self.b = {}
        # Dropout placeholder
        self.drop_ph = tf.placeholder(tf.float32)
        init = tf.initializers.glorot_normal()
        count_lay = 0
        # Weights encoder
        for inp, out in zip(self.lay_shape[:-1], self.lay_shape[1:]):
            w_name = "w"+str(count_lay)
            b_name = "b"+str(count_lay)
            self.w[w_name] = tf.get_variable(w_name, shape=[inp, out],
                                             initializer=init)
            self.b[b_name] = tf.get_variable(b_name, shape=[out],
                                             initializer=init)
            count_lay += 1

        # Weights decoder (reverse encoding)
        r_lay_shape = list(reversed(self.lay_shape))
        for inp, out in zip(r_lay_shape[:-1], r_lay_shape[1:]):
            w_name = "w"+str(count_lay)
            b_name = "b"+str(count_lay)
            self.w[w_name] = tf.get_variable(w_name, shape=[inp, out],
                                             initializer=init)
            self.b[b_name] = tf.get_variable(b_name, shape=[out],
                                             initializer=init)
            count_lay += 1
        # Layers
        self.layers = {}
        self.layers["hid0"] = tf.placeholder(tf.float32,
                                             shape=[None, in_shape],
                                             name="in_data")
        # Placeholder for the output data
        self.y_ph = tf.placeholder(tf.float32, shape=[None, in_shape],
                                   name="out_data")
        # Encoder output is from enc_node and enc_node + 1
        enc_node = len(self.lay_shape) - 2
        out_node = 2*len(self.lay_shape) - 2
        for i in range(out_node - 1):
            lay_name = "hid" + str(i)
            lay_name1 = "hid" + str(i + 1)
            w_i = "w" + str(i)
            b_i = "b" + str(i)
            self.layers[lay_name1] = tf.matmul(self.layers[lay_name],
                                               self.w[w_i]) + self.b[b_i]
            if i != enc_node:  # Encoder output won't relu or dropout
                self.layers[lay_name1] = tf.nn.relu(self.layers[lay_name1])
                self.layers[lay_name1] = tf.nn.dropout(self.layers[lay_name1],
                                                       rate=self.drop_ph)
        # Output layer
        i += 1
        lay_name = "hid" + str(i)
        w_i = "w" + str(i)
        b_i = "b" + str(i)
        self.layers["out"] = tf.matmul(self.layers[lay_name], self.w[w_i]) + \
            self.b[b_i]
        # Change names to encoder node
        # It appears at the end of the dictionary; doesn't matter
        self.layers["enc"] = self.layers.pop("hid" + str(enc_node + 1))
        return

    def fit(self, X, y=None, epoch=1000, batch_size=64, verbose=0,
            patience=0, min_delta=0.0001):
        """
        Method to fit the autoencoder.

        - X (2-dimensional numpy array): Input data to fit the autoencoder
        - y (1-dimensional numpy array): Output data if you want to make
                                         a denoising autoencoder.
                                         Default: None (simple autoencoder)
        - epoch (int): Number of full algorithm iterations
        - batch_size (int): Samples per weights update
        - verbose (int): Whether to print loss value each iteration
        - patience (int): If > 0, activates early stopping. Number of steps of
                          non-impoving TRAIN LOSS function after which
                          training will stop.
        - min_delta (float): Minimum change in loss to qualify as an
                             improvement.
        """
        # Non-denoising Autoencoder
        if y is None:
            y = X
        if X.shape[1] != y.shape[1]:
            print("X and y must have the same number of variables")
            return

        if min_delta < 0.0:
            min_delta = 0.0

        self.gen_network(X.shape[1])

        # Loss computation
        self.loss = tf.reduce_sum(tf.square(self.y_ph - self.layers["out"]))
        if self.l1 > 0:
            self.l1 = tf.constant(self.l1, tf.float32)
            reg1 = tf.zeros(1, tf.float32)
            for w_ptr in self.w:
                reg1 += tf.reduce_mean(tf.math.abs(self.w[w_ptr]))
            self.loss += self.l1*reg1
        if self.l2 > 0:
            self.l2 = tf.constant(self.l2, tf.float32)
            reg2 = tf.zeros(1, tf.float32)
            for w_ptr in self.w:
                reg2 += tf.reduce_mean(tf.math.abs(self.w[w_ptr]))
            self.loss += self.l2*reg2

        # Optimization part
        self.optimizer = tf.train.AdamOptimizer()
        self.train = self.optimizer.minimize(self.loss)

        # Create iterator
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        it = dataset.repeat().batch(batch_size)
        it = it.make_initializable_iterator()

        # Init global variables and iterator
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.sess.run(it.initializer)

        num_batch = X.shape[0]//batch_size

        # Tensorflow complains if we call get_next() inside the training
        # loop, so we create a variable to handle it
        next_batch = it.get_next()

        # Tracks the previous loss value for early stopping
        train_loss1 = 1.0e20
        patience_cnt = 0  # Counter for early stopping

        for e in range(epoch):
            train_loss = 0
            for b in range(num_batch):
                X_batch, y_batch = self.sess.run(next_batch)
                self.sess.run(self.train,
                              feed_dict={'in_data:0': X_batch,
                                         self.y_ph: y_batch,
                                         self.drop_ph: self.dropout})
                train_loss += self.sess.run(self.loss,
                                            feed_dict={'in_data:0': X_batch,
                                                       self.y_ph: y_batch,
                                                       self.drop_ph: 0.0})
            if verbose > 0:
                print("Epoch {} loss {}".format(e, train_loss))

            # Early stopping
            if patience > 0:
                if e > 0 and train_loss1 - train_loss > min_delta:
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt > patience:
                    break
                train_loss1 = train_loss
        return

    def predict(self, X):
        return self.sess.run(self.layers["out"],
                             feed_dict={'in_data:0': X,
                                        self.drop_ph: 0.0})

    def encode(self, X):
        return self.sess.run(self.layers["enc"],
                             feed_dict={'in_data:0': X,
                                        self.drop_ph: 0.0})

    def decode(self, X):
        return self.sess.run(self.layers["out"],
                             feed_dict={self.layers["enc"]: X,
                                        self.drop_ph: 0.0})

    def score(self, X, y=None):
        if y is None:
            y = X
        y_pred = self.predict(X)
        # The minus sign is for making it compatible with scikit-learn
        # as it always maximizes, and this is the mean squared error
        return - np.mean(np.sum(np.square(y - y_pred), axis=0))

    def get_params(self, deep=True):
        return {
            "code_len": self.code_len, "shape": self.shape,
            "dropout": self.dropout, "l1": self.l1, "l2": self.l2}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
