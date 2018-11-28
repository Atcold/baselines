import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from math import sqrt

# Constants
STATE_C = 3
STATE_H, STATE_W = 117, 24
STATE_D = 4

def traffic_model(n_cond):

    activation = tf.nn.leaky_relu

    def encode_s(state):
        # TODO add scaling?
        h = tf.reshape(state, [-1, n_cond * STATE_D])
        kwargs = dict(nh=256, init_scale=sqrt(2))
        h = activation(fc(h, 'enc_s_fc1', **kwargs))
        h = activation(fc(h, 'enc_s_fc2', **kwargs))
        return fc(h, 'enc_s_fc3', nh=10752, init_scale=sqrt(2))

    def encode_c(context):
        h = tf.reshape(context, [-1, n_cond * STATE_C, STATE_H, STATE_W])
        # h = tf.cast(h, tf.float32) / 255
        # nf: nb_features, rf: receptive_field
        # h = tf.transpose(h, perm=[0, 2, 3, 1])  # because CPU version does not support NCHW
        # p = tf.constant([[0, 0,], [1, 1], [1, 1], [0, 0]])
        p = tf.constant([[0, 0,], [0, 0], [1, 1], [1, 1]])
        kwargs = dict(rf=4, stride=2, init_scale=sqrt(2), data_format='NCHW', pad='VALID')
        h = activation(conv(tf.pad(h, p), 'enc_i_c1', nf=64, **kwargs))
        h = activation(conv(tf.pad(h, p), 'enc_i_c2', nf=128, **kwargs))
        h = activation(conv(tf.pad(h, p), 'enc_i_c3', nf=256, **kwargs))
        return conv_to_fc(h)

    def project(h):
        return fc(h, 'prj', nh=256, init_scale=sqrt(2))

    def mlp(h):
        activation = tf.nn.relu
        kwargs = dict(nh=256, init_scale=sqrt(2))
        h = activation(fc(h, 'mlp_fc1', **kwargs))
        h = activation(fc(h, 'mlp_fc2', **kwargs))
        h = activation(fc(h, 'mlp_fc3', **kwargs))
        return fc(h, 'mlp_fc4', **kwargs)

    def encode(x):
        state = x[:, :, :STATE_D]
        context = x[:, :, STATE_D:]
        emb_s = encode_s(state)  # 10752 dimensional
        emb_c = encode_c(context)
        return project(emb_s + emb_c)  # 256 dimensional

    def network_fn(X):
        return mlp(encode(X)), None

    return network_fn

if __name__ == '__main__':

    import numpy as np
    from baselines.common.input import observation_placeholder
    from baselines.common.tf_util import adjust_shape, get_session
    from gym import spaces
    from baselines.common import tf_util

    # Parameters
    n_cond = 20

    # Spaces definition
    observation_space = spaces.Box(
        low=-1, high=1, shape=(n_cond, STATE_D + STATE_C * STATE_H * STATE_W), dtype=np.float32
    )

    # Fake data creation
    observation = np.random.normal(size=(1, n_cond, STATE_D + STATE_C * STATE_H * STATE_W))

    model = traffic_model(n_cond=n_cond)
    X = observation_placeholder(observation_space, batch_size=None)
    embedding, _ = model(X)

    sess = tf_util.get_session()
    tf.global_variables_initializer().run(session=sess)

    feed_dict = {X: adjust_shape(X, observation)}

    emb_value = sess.run(embedding, feed_dict)
    print(emb_value.shape)
