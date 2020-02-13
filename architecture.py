import numpy as np
import tensorflow as tf

#this function selects prob distribution over actions
from baselines.common.distributions import make_pdtype

def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs, filters=filters, 
        kernel_size=kernel_size, strides=(strides), activation=tf.nn.relu, 
        kernel_initializer=tf.orthogonal_initializer(gain=gain))
def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
    return tf.layers.dense(inputs=inputs, units=units, activation = activation_fn, 
        kernel_initializer=tf.orthogonal_initializer(gain=gain))

class A2CPolicy(object):
    def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse=False):
        #kernel initialization
        gain = np.sqrt(2)

        #based on action space we will select our probability distribution type
        # to distribiute actions in our stochastic policy (will use DiagGaussianPdType)
        self.pdtype = make_pdtype(action_space)

        height, weight, channel = ob_space.shape
        ob_shape = (height, weight, channel)

        #create the input placeholder
        inputs_ = tf.placeholder(tf.float32, [None, *ob_shape], name="input")

        #Normalize our images
        scaled_images = tf.cast(inputs_, tf.float32) / 255.

        #Build our model
        #3-Layer CNN for spatial dependencies
        #Temporal dependencies handled by frame stacking
        #Single FC Layer
        #FC Layer for our Policy
        #FC Layer for our Value
        with tf.variable_scope("model", reuse=reuse):
            conv1 = conv_layer(scaled_images, 32, 8, 4, gain)
            conv2 = conv_layer(conv1, 64, 4, 2, gain)
            conv3 = conv_layer(conv2, 64, 3, 1, gain)
            flatten1 = tf.layers.flatten(conv3)
            fc_common = fc_layer(flatten1, 512, gain=gain)

            #build a probability distribution over actions and policy logits
            self.pd, self.pi = self.pdtype.pdfromlatent(fc_common, init_scale=0.01)

            #calculate the v(s)
            vf = fc_layer(fc_common, 1, activation_fn=None)[:, 0]
        
        #?? Given error without setting an intial state.
        self.initial_state = None

        #Take an action of the distribution (Stochastic policy, so we don't always take argmax of actions)
        a0 = self.pd.sample()

        #Function to take a step returns actions to take & V(s)
        def step(state_in, *_args, **_kwargs):
            #grab the action and V(S) from our session
            action, value = sess.run([a0, vf], {inputs_:state_in})
            return action, value
        
        def value(state_in, *_args, **_kwargs):
            return sess.run(vf, {inputs_:state_in})

        def select_action(state_in, *_args, **_kwargs):
            return sess.run(a0, {inputs_:state_in})

        self.inputs_ = inputs_
        self.vf = vf
        self.step = step
        self.value = value
        self.select_action = select_action



