import os
import time
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
import cv2
import matplotlib.pyplot as plt

#To calculate our cross entropy
from baselines.a2c.utils import cat_entropy, mse
from utilities import make_path, find_trainable_variables, discount_with_dones

from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner

class Model(object):
    #We're going to create the step_model
    #We're going to create the train_model
    #We're going to setup training
    #We're going to setup saving/loading

    def __init__(self, policy, ob_space, action_space, nenvs, nsteps, ent_coeff, vf_coeff, max_grad_norm):
        sess = tf.get_default_session()

        #Define placeholders
        actions_ = tf.placeholder(tf.int32, [None], name="actions_")
        advantages_ = tf.placeholder(tf.float32, [None], name="advantages_")
        rewards_ = tf.placeholder(tf.float32, [None], name="rewards_")
        lr_ = tf.placeholder(tf.float32, name="learning_rate_")

        #Create our two models here
        #take one step for each environment
        step_model = policy(sess, ob_space, action_space, nenvs, 1, reuse=False)
        #take number of steps * number of environments for total steps
        train_model = policy(sess, ob_space, action_space, nenvs*nsteps, nsteps, reuse=True)

        #calculate the loss
        #Note: in the future we can add clipped Loss to control the step size of our parameter updates. 
        #This can lead to better convergence *Using PPO*
        #Recall that Total Loss =  PolicyGradientLoss - Entropy*EntropyCoeff + Value*ValueCoeff

        #output loss -log(policy)
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(Logits=train_model.pi, Labels=actions_,)

        #1/n * sum(A(s,a) * -logpi(a|s))
        pg_loss = tf.reduce_mean(advantages_ * neglogpac)

        #value loss
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), rewards_))

        #entropy
        entropy = tf.reduce_mean(train_model.pd.entropy())

        #total loss
        loss = pg_loss - (entropy * ent_coeff) + (vf_loss * vf_coeff)

        #Update the parameters using the loss we've just calculated
        #Grab model params
        params = find_trainable_variables("model")

        #Calculate gradients. *We'll want to zip our parameters w/ our gradients
        grads = tf.gradients(loss, params)

        if max_grad_norm is not None:
            #Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = List(zip(grads, params))

        #build our trainer
        trainer = tf.train.RMSPropOptimizer(Learning_rate = lr_, decay=0.99, epsilon=1e-5)
        #Backprop
        _train = trainer.apply_gradients(grads)

        def train(states_in, actions, returns, values, lr):
            #here we calculate advantage A(s, a) = R+yV(s') - V(s)
            #Returns = R+yV(S')
            advantages = returns-values

            td_map = {
                train_model.inputs_:states_in,
                actions_:actions,
                advantages_:advantages,
                rewards_:returns, #Recall we bootstrap "real" value since we're learning 1 step at a time. (not episode)
                lr_:lr
            }

            policy_loss, value_loss, policy_entropy, _ = sess.run([pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            saver = tf.train.Saver()
            saver.save(sess, save_path)
        
        def load(load_path):
            saver = tf.train.Saver()
            saver.restore(sess, load_path)
        
        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(AbstractEnvRunner):
    #We use this object to make mini-batch of experiences

    def __init__(self, env, model, nsteps, total_timesteps, gamma, lam):
        super().__init__(env = env, model = model, nsteps = nsteps)

        #Discount factor
        self.gamma = gamma
        #Lambda value in GAE
        self.lam = lam
        #total timesteps taken
        self.total_timesteps = total_timesteps
    
    def run(self):
        #initialize lists for mini batch experiences
        mb_obs, mb_actions, mb_rewards, mb_values, mb_dones = [], [], [], [], []

        #for n in range number of steps
        for n in range(self.nsteps):
            #given obs take action and value V(s)
            #Leveraging AbstractEnvRunner self.obs[:] = env.reset()
            actions, values = self.model.step(self.obs, self.dones)

            #append observations to mb
            mb_obs.append(np.copy(self.obs))

            #append the actions taken into the mb
            mb_actions.append(actions)

            #append the values calculated into the mb
            mb_values.append(values)

            #append the dones into the mb
            mb_dones.append(self.dones)

            #take actions in env
            self.obs[:], rewards, self.dones, _ = self.env.step(actions)

            mb_rewards.append(rewards)
        
        #We want to feed these batches into our network as numpy arrays
        mb_obs = np.asarray(mb_obs, dtype=np.uint8)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs)


        #GAE (Generalized Advantage Estimation)
        #We are generalizing our traditional advantage function
        #create mb_returns and mb_advantages
        #returns is reward + gamma * value of next state OR Advantage + value of state
        #

        mb_returns = np.zeros_like(mb_rewards)
        mb_advantages = np.zeros_like(mb_rewards)
        lastgaelam = 0

        #from last step to first step
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                #if state is done, nextnontterminal = 0; delta = R - V(s)
                #else delta = R + gamma * V(s(t+1))
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mv_values[t+1]

            delta = mv_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

            mb_advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal*lastgaelam

        #Returns
        mb_returns = mb_advantages + mb_values

        return map(sf01, (mb_obs, mb_actions, mb_returns, mb_values))

def sf01(arr):
    #swap and then flatten axes 0 and 1
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def learn(policy, env, nsteps, total_timesteps, gamma, lam, vf_coeff, ent_coeff, lr, max_grad_norm, log_interval):
    #Combines all elements to train our agent

    #train each mini- batch for 4 epochs
    noptepochs = 4
    #Break our large batch that we potentially can't hold in our program into mini batches
    nminibatches = 8

    #get the nb of env
    nenvs = env.num_envs
    #get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    #calculate the batch size
    batch_size = nenvs * nsteps
    batch_train_size = batch_size //nminibatches

    assert batch_size % nminibatches == 0

    model = Model(policy=policy, ob_space=ob_space, action_space=ac_space, 
                nenvs=nenvs, nsteps=nsteps, ent_coeff=ent_coeff, 
                vf_coeff=vf_coeff, max_grad_norm=max_grad_norm)

    #Load model option if we want to continue training


    #Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, total_timesteps=total_timesteps, gamma=gamma, lam=lam)

    #total timer init
    tfirststart = time.time()

    for update in range(1, total_timesteps//batch_size+1):
        #mb timer
        tstart = time.time()

        obs, actions, returns, values = runner.run()

        #for ea minibatch, calculate the and append it. 
        mb_losses = []
        total_batches_train = 0

        #index of ea element of batch_size
        indices = np.arange(batch_size)

        for _ in range(noptepochs):
            #randomize indexes *randomize our experiences* *helps fight overfitting*
            np.random.shuffle(indices)

            #go from 0 to batch size with batch train size step
            for start in range(0, batch_size, batch_train_size):
                end = start + batch_train_size
                mbinds = indices[start:end]
                slices = (arr[mbinds] for arr in (obs, actions, returns, values))
                #determine and append loss of our minibatch
                mb_losses.append(model.train(*slices, lr))

        lossvalues = np.mean(mb_losses, axis=0)

        tnow = time.time()

        #calculate fps
        fps = int(batch_size / (tnow-tstart))

        if update % log_interval == 0 or update == 1:
            #Log all elements we'd like to track
            #we want our expalined variance to reach closer & closer to 1 *perfect*
            ev = explained_variance(values, returns)

            logger.record_tabular("serial_timesteps", update*nsteps)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*batch_size)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_loss", float(lossvalues[0]))
            logger.record_tabular("policy_entropy", float(lossvalues[2]))
            logger.record_tabular("value_loss", float(lossvalues[1]))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time elapsed", float(tnow - tfirststart))

            savepath = "./models/" + str(update) + "/model.ckpt"
            model.save(savepath)
            print('Saving to', savepath)

            #test our agent with 3 trials and mean the score
            #this will help us see if our agent is imporving
            test_score = testing(model)

            logger.record_tabular("Mean score test level", test_score)
            logger.dump_tabular()

    env.close()








