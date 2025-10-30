# pinn_architecture.py
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras import activations as keras_activations # Explicit import
import numpy as np

class RandomWeightFactorizedDense(layers.Layer):
    def __init__(self, units, activation=None, s_mean=0.5, s_std=0.1, **kwargs):
        super(RandomWeightFactorizedDense, self).__init__(**kwargs)
        self.units = units
        self.activation_fn = keras_activations.get(activation)
        self.s_mean = s_mean
        self.s_std = s_std

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.V = self.add_weight(
            name="V",
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True
        )
        self.s = self.add_weight(
            name="s",
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(mean=self.s_mean, stddev=self.s_std),
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        super(RandomWeightFactorizedDense, self).build(input_shape)

    def call(self, inputs):
        scaling = tf.exp(self.s)
        W = self.V * scaling
        output = tf.linalg.matmul(inputs, W) + self.bias
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

    def get_config(self):
        config = super(RandomWeightFactorizedDense, self).get_config()
        config.update({
            "units": self.units,
            "activation": keras_activations.serialize(self.activation_fn),
            "s_mean": self.s_mean,
            "s_std": self.s_std
        })
        return config

class PirateNetBlock(layers.Layer):
    def __init__(self, hidden_dim, activation_fn_str='tanh', RWFactorized=True, **kwargs):
        super(PirateNetBlock, self).__init__(**kwargs)
        self.activation = keras_activations.get(activation_fn_str)
        self.RWFactorized = RWFactorized
        self.hidden_dim = hidden_dim # Store for get_config

        if RWFactorized:
            self.dense_f = RandomWeightFactorizedDense(hidden_dim, activation=self.activation)
            self.dense_g = RandomWeightFactorizedDense(hidden_dim, activation=self.activation)
            self.dense_h = RandomWeightFactorizedDense(hidden_dim, activation=self.activation)
        else:
            self.dense_f = layers.Dense(hidden_dim, activation=self.activation,
                                        kernel_initializer='glorot_normal', bias_initializer='zeros')
            self.dense_g = layers.Dense(hidden_dim, activation=self.activation,
                                        kernel_initializer='glorot_normal', bias_initializer='zeros')
            self.dense_h = layers.Dense(hidden_dim, activation=self.activation,
                                        kernel_initializer='glorot_normal', bias_initializer='zeros')
        self.alpha = self.add_weight(name='alpha', shape=(), initializer='zeros', trainable=True)

    def call(self, x, U, V):
        f = self.dense_f(x)
        z1 = f * U + (1 - f) * V
        g = self.dense_g(z1)
        z2 = g * U + (1 - g) * V
        h = self.dense_h(z2)
        return self.alpha * h + (1 - self.alpha) * x

    def get_config(self):
        config = super(PirateNetBlock, self).get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "activation_fn_str": keras_activations.serialize(self.activation), # Save the resolved activation
            "RWFactorized": self.RWFactorized
        })
        return config

class PirateNet(Model):
    def __init__(self, input_dim, output_dim, m=128, s_init_val=1.0, L=3,
                 activation_fn_str='tanh', RWFactorized=True, **kwargs):
        super(PirateNet, self).__init__(**kwargs)
        self.input_dim_val = input_dim # Store with different name to avoid Keras conflict
        self.output_dim_val = output_dim
        self.m_val = m
        self.s_init_val = s_init_val
        self.L_val = L
        self.activation = keras_activations.get(activation_fn_str)
        self.RWFactorized_val = RWFactorized

        B_np = np.random.randn(self.m_val, self.input_dim_val) * self.s_init_val
        self.B_matrix = self.add_weight( # Renamed self.B to self.B_matrix
            name="B_matrix",
            shape=(self.m_val, self.input_dim_val),
            initializer=tf.keras.initializers.Constant(B_np),
            trainable=False
        )
        self.embedding_dim = 2 * self.m_val

        if self.RWFactorized_val:
            self.U_layer = RandomWeightFactorizedDense(self.embedding_dim, activation=self.activation)
            self.V_layer = RandomWeightFactorizedDense(self.embedding_dim, activation=self.activation)
        else:
            self.U_layer = layers.Dense(self.embedding_dim, activation=self.activation,
                                        kernel_initializer='glorot_normal', bias_initializer='zeros')
            self.V_layer = layers.Dense(self.embedding_dim, activation=self.activation,
                                        kernel_initializer='glorot_normal', bias_initializer='zeros')

        self.blocks = [PirateNetBlock(self.embedding_dim, activation_fn_str=activation_fn_str, # Pass string here
                                      RWFactorized=self.RWFactorized_val)
                       for _ in range(self.L_val)]

        if self.RWFactorized_val:
            self.final_layer = RandomWeightFactorizedDense(self.output_dim_val, activation=None)
        else:
            self.final_layer = layers.Dense(self.output_dim_val, kernel_initializer='glorot_normal', bias_initializer='zeros')

    def call(self, x):
        projection = tf.linalg.matmul(x, tf.transpose(self.B_matrix))
        embed = tf.concat([tf.cos(projection), tf.sin(projection)], axis=-1)
        U = self.U_layer(embed)
        V = self.V_layer(embed)
        x_rep = embed
        for block in self.blocks:
            x_rep = block(x_rep, U, V)
        return self.final_layer(x_rep)

    def get_config(self):
        # To ensure correct serialization, use the stored init args
        config = super(PirateNet, self).get_config()
        config.update({
            "input_dim": self.input_dim_val,
            "output_dim": self.output_dim_val,
            "m": self.m_val,
            "s_init_val": self.s_init_val,
            "L": self.L_val,
            "activation_fn_str": keras_activations.serialize(self.activation),
            "RWFactorized": self.RWFactorized_val
        })
        return config

if __name__ == '__main__':
    print("Testing PirateNet architecture...")
    # Example: input_dim = spatial_dims (2 for x,y) + latent_dims (N_g) + time_dim (1)
    test_input_dim = 2 + 2 + 1 # Example: N_g=2
    h_model_test = PirateNet(input_dim=test_input_dim, output_dim=1, m=64, s_init_val=5., L=3,
                             activation_fn_str='tanh', RWFactorized=True)
    x_input_test = tf.random.normal((10, test_input_dim))
    output_test = h_model_test(x_input_test)
    h_model_test.summary(line_length=120)
    print(f"Output shape: {output_test.shape}")

    # Test serialization
    config = h_model_test.get_config()
    print("\nModel Config:", config)
    # new_model = PirateNet.from_config(config) # Requires custom_objects for custom layers if not registered
    # print("\nModel successfully recreated from config.") # (Needs custom object registration for this to work directly)

    print("PirateNet architecture test complete.")