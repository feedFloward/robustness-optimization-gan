from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from typing import List

class Generator:
    def __init__(self, hidden_units : List[int], latent_dim : int, output_dim : int):
        self.latent_dim = latent_dim
        self.model = Sequential()
        self.model.add(Input(shape=(latent_dim,)))
        for num_units in hidden_units:
            self.model.add(Dense(num_units, activation= 'relu'))
        self.model.add(Dense(output_dim, activation='tanh'))

    def sample_latent_vector(self, batch_size):
        return tf.random.normal(shape=(batch_size, self.latent_dim))

    def generate_samples(self, num_samples):
        return self.model(self.sample_latent_vector(num_samples))

class Discriminator:
    def __init__(self, hidden_units : List[int], input_dim : int):
        self.model = Sequential()
        self.model.add(Input(shape=(input_dim,)))
        for num_units in hidden_units:
            self.model.add(Dense(num_units, activation= 'relu'))
        self.model.add(Dense(1, activation= 'sigmoid'))

class GAN(Model):
    def __init__(self, hidden_units_gen, hidden_units_disc, latent_dim, gen_output_dim):
        super(GAN, self).__init__()
        self.generator = Generator(
            hidden_units= hidden_units_gen,
            latent_dim= latent_dim,
            output_dim= gen_output_dim
        )
        self.discriminator = Discriminator(
            hidden_units= hidden_units_disc,
            input_dim= gen_output_dim
        )

    def compile(self, lr_disc, lr_gen):
        super(GAN, self).compile()
        self.d_optimizer = Adam(learning_rate= lr_disc)
        self.g_optimizer = Adam(learning_rate= lr_gen)
        self.loss_fn = BinaryCrossentropy(from_logits= False)

    def train_step(self, real_input):
        if isinstance(real_input, tuple):
            real_input = real_input[0]
        batch_size = tf.shape(real_input)[0]
        latent_vector = self.generator.sample_latent_vector(batch_size)

        fake_input = self.generator.model(latent_vector)


        #später einfügen: add random noise to labels

        with tf.GradientTape() as tape:
            predictions_for_real = self.discriminator.model(real_input)
            predictions_for_fake = self.discriminator.model(fake_input)
            d_loss = self._discriminator_loss(predictions_for_real, predictions_for_fake)
        grads = tape.gradient(d_loss, self.discriminator.model.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.model.trainable_weights)
        )


        latent_vector = self.generator.sample_latent_vector(batch_size)

        with tf.GradientTape() as tape:
            predictions = self.discriminator.model(self.generator.model(latent_vector))
            g_loss = self._generator_loss(predictions)
        grads = tape.gradient(g_loss, self.generator.model.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.model.trainable_weights)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

    def _discriminator_loss(self, real_input, fake_input):
        real_loss = self.loss_fn(tf.ones_like(real_input), real_input)
        fake_loss = self.loss_fn(tf.zeros_like(fake_input), fake_input)
        return real_loss + fake_loss

    def _generator_loss(self, fake_input):
        return self.loss_fn(tf.ones_like(fake_input), fake_input)