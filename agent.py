import tensorflow as tf
import numpy as np
import gym
import math
import os

import model
import architecture as policies
import sonic_env as env

from baselines.common.evec_env.subproc_vec_env import SubprocVecEnv

def main():
    config = tf.ConfigProto()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    #Allow GPU Memory Growth
    config.gpu_options.allow_growth = True


    #note: SubprocVecEnv places all our environments in a vector which will allow for us to run them simultaneously 
    with tf.Session(config=config):
        #Call the learn function with all the required A2C Policy Params
        model.learn(
            policy=policies.A2CPolicy,
            env=SubprocVecEnv([env.make_train_0, env.make_train_1, env.make_train_2, env.make_train_3, 
                env.make_train_4, env.make_train_5, env.make_train_6, env.make_train_7, env.make_train_8, 
                env.make_train_9, env.make_train_10, env.make_train_11, env.make_train_12]),
            nsteps=2048,
            total_timesteps=10000000,
            gamma=0.99,
            lam=0.95,
            vf_coeff=0.5,
            ent_coeff=0.01,
            lr = lambda _: 2e-4,
            max_grad_norm = 0.5, #Avoid big gradient steps 
            log_interval = 10 #print in our console every 10 weight updates
        )


if __name__ == '__main__':
    main()
