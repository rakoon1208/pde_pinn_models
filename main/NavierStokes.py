import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time


class PhysicsInformedNN:
    def __init__(self, x, y, t, u, v, layers):
        # Combine inputs into a single array
        X = np.concatenate([x, y, t], axis=1)

        self.lb = tf.constant(X.min(0), dtype=tf.float32)
        self.ub = tf.constant(X.max(0), dtype=tf.float32)

        self.X = tf.convert_to_tensor(X, dtype=tf.float32)
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)
        self.y = tf.convert_to_tensor(y, dtype=tf.float32)
        self.t = tf.convert_to_tensor(t, dtype=tf.float32)
        self.u = tf.convert_to_tensor(u, dtype=tf.float32)
        self.v = tf.convert_to_tensor(v, dtype=tf.float32)

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Physics-informed parameters
        self.lambda_1 = tf.Variable(0.0, dtype=tf.float32)
        self.lambda_2 = tf.Variable(0.0, dtype=tf.float32)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim, out_dim = size
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W, b = self.weights[-1], self.biases[-1]
        return tf.add(tf.matmul(H, W), b)

    def net_NS(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, t])
            psi_and_p = self.neural_net(tf.concat([x, y, t], axis=1))
            psi = psi_and_p[:, 0:1]
            p = psi_and_p[:, 1:2]

            # Velocities
            u = tape.gradient(psi, y)
            v = -tape.gradient(psi, x)

        # Gradients for Navier-Stokes
        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)

        v_t = tape.gradient(v, t)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)

        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)

        del tape

        f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + p_x - self.lambda_2 * (u_xx + u_yy)
        f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + p_y - self.lambda_2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss(self, x, y, t, u, v):
        u_pred, v_pred, _, f_u_pred, f_v_pred = self.net_NS(x, y, t)
        loss_data = tf.reduce_mean(tf.square(u - u_pred)) + tf.reduce_mean(tf.square(v - v_pred))
        loss_pde = tf.reduce_mean(tf.square(f_u_pred)) + tf.reduce_mean(tf.square(f_v_pred))
        return loss_data + loss_pde

    @tf.function
    def train_step(self, x, y, t, u, v):
        with tf.GradientTape() as tape:
            loss_value = self.loss(x, y, t, u, v)
        gradients = tape.gradient(loss_value, self.weights + self.biases + [self.lambda_1, self.lambda_2])
        self.optimizer.apply_gradients(zip(gradients, self.weights + self.biases + [self.lambda_1, self.lambda_2]))
        return loss_value

    def train(self, nIter, x, y, t, u, v):
        for it in range(nIter):
            loss_value = self.train_step(x, y, t, u, v)
            if it % 10 == 0:
                print(f"Iteration {it}, Loss: {loss_value.numpy():.4e}, λ1: {self.lambda_1.numpy():.4f}, λ2: {self.lambda_2.numpy():.4f}")

    def predict(self, x_star, y_star, t_star):
        u_pred, v_pred, p_pred, _, _ = self.net_NS(x_star, y_star, t_star)
        return u_pred.numpy(), v_pred.numpy(), p_pred.numpy()


# Example usage
if __name__ == "__main__":
    # Example data
    N_train = 1000
    x = np.random.rand(N_train, 1)
    y = np.random.rand(N_train, 1)
    t = np.random.rand(N_train, 1)
    u = np.sin(np.pi * x) * np.sin(np.pi * y)
    v = -np.sin(np.pi * x) * np.sin(np.pi * y)

    # Convert to tensors
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)
    v_tf = tf.convert_to_tensor(v, dtype=tf.float32)

    # Layers: Input (3) -> Hidden (20) -> Output (2)
    layers = [3, 20, 20, 20, 20, 2]

    # Initialize and train the model
    model = PhysicsInformedNN(x_tf, y_tf, t_tf, u_tf, v_tf, layers)
    model.train(1000, x_tf, y_tf, t_tf, u_tf, v_tf)
