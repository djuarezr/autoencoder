from mxnet import gluon


class autoencoder(gluon.HybridBlock):
    def __init__(self, input_shape, layers=[10], med_lay=3, drop=0.1):
        super(autoencoder, self).__init__()
        with self.name_scope():
            self.encoder = gluon.nn.HybridSequential('encoder_')
            with self.encoder.name_scope():
                for i in layers:
                    self.encoder.add(gluon.nn.Dense(i, activation='relu'))
                    if drop > 0:
                        self.encoder.add(gluon.nn.Dropout(rate=drop))
                self.encoder.add(gluon.nn.Dense(med_lay))

            self.decoder = gluon.nn.HybridSequential('decoder_')
            with self.decoder.name_scope():
                for i in reversed(layers):
                    if drop > 0:
                        self.decoder.add(gluon.nn.Dropout(rate=drop))
                    self.decoder.add(gluon.nn.Dense(i, activation='relu'))
                self.decoder.add(gluon.nn.Dense(input_shape))

    def hybrid_forward(self, F, x):
        self.encoded = self.encoder(x)
        self.decoded = self.decoder(self.encoded)
        return self.decoded, self.encoded
