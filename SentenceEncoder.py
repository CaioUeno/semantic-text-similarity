import numpy as np
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Bidirectional,
    BatchNormalization,
    Embedding,
    Dot,
)
from tensorflow.keras.losses import MAE
from tensorflow.math import maximum, divide_no_nan, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import string as tfstring


class SentenceEncoder(object):

    """
    Sentence Embedder.
    """

    def __init__(
        self,
        vocab: list,
        max_tokens: int = None,
        max_sentence_length: int = 64,
        embedding_size: int = 32,
    ):

        self.max_tokens = max_tokens
        self.max_sentence_length = max_sentence_length
        self.embedding_size = embedding_size
        self.base_embedder = self.get_base_embedder(vocab)
        self.trainable_model = self.get_trainable_model()

    def get_base_embedder(self, vocab):

        # text-to-index-list layer
        encoder = TextVectorization(
            max_tokens=self.max_tokens,
            output_mode="int",
            output_sequence_length=self.max_sentence_length,
            vocabulary=vocab,
        )

        # embedder model
        # text input layer
        input_layer = Input(shape=(1,), dtype=tfstring)

        # text to index layer
        vectorize_layer = encoder(input_layer)

        # embedding layer
        embedding_layer = Embedding(
            input_dim=len(encoder.get_vocabulary()), output_dim=32, mask_zero=True
        )(vectorize_layer)

        # bidirectional lstm layer
        bi_lstm_layer = Bidirectional(
            LSTM(32, name="lstm-layer"), name="bidirectional-layer"
        )(embedding_layer)

        # normalization layer
        norm_layer = BatchNormalization()(bi_lstm_layer)

        # final embedding layer
        embedding = Dense(self.embedding_size, name="embedding-layer")(norm_layer)

        return Model(inputs=input_layer, outputs=embedding)

    def get_trainable_model(self):

        # input for two distinct examples
        input_example_1 = Input(shape=(1,), dtype=tfstring, name="input-example-1")
        input_example_2 = Input(shape=(1,), dtype=tfstring, name="input-example-2")

        # layer to simulate inter product between two vectors (cosine similarity)
        # note the normalize param set to True
        dot_layer = Dot(axes=1, normalize=True)(
            [
                self.base_embedder(input_example_1),
                self.base_embedder(input_example_2),
            ]
        )

        trainable_model = Model(
            inputs=[
                input_example_1,
                input_example_2,
            ],
            outputs=dot_layer,
        )
        
        def reloss(y_true, y_pred):

            """
            Custom loss to not penalize when the prediction is negative (dissimilar) and true label is 0.
            """

            loss_filter = maximum(y_true, y_pred)
            loss_filter = divide_no_nan(loss_filter, loss_filter) # normalize any positive value to 1
            return multiply(loss_filter, MAE(y_true, y_pred))

        
        trainable_model.compile(loss=reloss, optimizer="adam")

        return trainable_model

    def fit(self, X, y, epochs: int = 35, batchsize: int = 128):

        # note: here, X is actually a list of two text arrays -> [sentences, sentences]
        return self.trainable_model.fit(X, y, batch_size=batchsize, epochs=epochs)

    def fit_generator(self, generator, epochs: int = 35, steps_per_epoch: int = 128):

        # note: here, X is a generator
        return self.trainable_model.fit(generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def encode(self, X):

        # note: here, X is just one sentences array
        return self.base_embedder.predict(np.array(X))
