import tensorflow as tf


class autoencoder:
    def __init__(self,
                 input_shape,
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

        # Avoid problems re-using names of the variables
        tf.reset_default_graph()

        self.input_shape = input_shape
        self.shape = shape
        self.code_len = code_len
        self.dropout = dropout
        self.l1 = l1
        self.l2 = l2

        self.lay_shape = [self.input_shape] + self.shape + [self.code_len]
        # Weights and biases
        self.w = {}
        self.b = {}
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
                                             shape=[None, input_shape],
                                             name="in_data")
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
            if i != enc_node:  # Encoder output won't relu
                self.layers[lay_name1] = tf.nn.relu(self.layers[lay_name1])
                # Encoder output will not dropout
                if self.dropout > 0 and i != enc_node:
                    self.layers[lay_name1] = tf.nn.dropout(
                        self.layers[lay_name1], rate=dropout
                    )
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

        # Loss computation
        self.loss = tf.reduce_mean(tf.square(self.layers["out"] -
                                             self.layers["hid0"]))
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
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.loss)
        return

    def fit(self, x, epoch=1000, batch_size=64, verbose=0):
        """
        Method to fit the autoencoder.

        - x (2-dimensional numpy array): Input data to fit the autoencoder
        - epoch (int): Number of full algorithm iterations
        - batch_size (int): Samples per weights update
        - verbose (int): Whether to print loss value each iteration
        """
        self.dataset = tf.data.Dataset.from_tensor_slices((x))
        self.it = self.dataset.repeat().batch(batch_size)
        self.it = self.it.make_initializable_iterator()

        # Init global variables and iterator
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.sess.run(self.it.initializer)

        num_batch = x.shape[0]//batch_size

        # Tensorflow complains if we call get_next() inside the training
        # loop, so we create a variable to handle it
        next_batch = self.it.get_next()
        for e in range(epoch):
            train_loss = 0
            for b in range(num_batch):
                x_batch = self.sess.run(next_batch)
                self.sess.run(self.train, feed_dict={'in_data:0': x_batch})
                train_loss += self.sess.run(self.loss,
                                            feed_dict={"in_data:0": x_batch})
            if verbose > 0:
                print("Epoch {} loss {}".format(e, train_loss))
        return

    def predict(self, x):
        return self.sess.run(self.layers["out"], feed_dict={"in_data:0": x})

    def encode(self, x):
        return self.sess.run(self.layers["enc"], feed_dict={"in_data:0": x})

    def __del__(self):
        self.sess.close()
        return
