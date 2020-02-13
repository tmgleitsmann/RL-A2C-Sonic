NOTE: This code is from Thomas Simonini's Reinforcement Learning Course. I do not take credit for it.
https://simoninithomas.github.io/Deep_reinforcement_learning_Course/

*AS WELL AS PIP INSTALLING THE REQUIREMENTS, YOU WILL NEED MUJOCO AS IT IS A REQUIREMENT FOR BASELINES*
http://www.mujoco.org/ you can get a license for it here. Ensure your installation is correct before pip installing the requirements.txt

THIS HAS NOT BEEN TRAIN NOR TESTED YET! Can use the code to understand the algorithm behind A2C. Can also modify the loss calcuation to produce a PPO implementation that might converge better.

I followed along Thomas Simonini's courses because I wanted to see how A2C and PPOs worked. Putting these concepts to code helps me solidiy my understanding. The purpose of the code for now is to demonstrate my understand of Advantage Actor Critic methods in Reinforcement Learning.

A major caveat of mine is that this code requires both a CUDA GPU to train with and a steam account with Sonic the Hedgehog.
This is incompatible with my Jetson Tx2 so I'll need to borrow a friend's PC to train and test the model.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In the project we have 4 python files defined, but the program execution will start from agent.py

Agent.py
-Create our tensorflow session and call on our learn function (specified in model.py) with the parameters...
	-policies.A2CPolicy (specified in architecture.py) defines our model architecture and allows us to take an action & calculate V(S)
	-environemnts specified by sonic_env.py. This creates a vector of subprocesses for all 13 sonice environments.
	-number of steps we want to take per environment
	-total timesteps
	-gamma (discount factor)
	-lambda used for GAE
	-Value and Entropy Coefficients
	-Learning Rate
	-max_grad_norm : help us control our gradient steps

Model.py
-Holds the definition of our Model class, Runner class and learn function.

-Learn trains our agent and is located inside of model.py. It is responible for-
	-Creating our model based off the architecture passed from Agent.py
	-Creating a Runner object that will take steps through our environments. Return observations, actions, returns(QVal) and values V(S)
	-Use these returned values from Runner to train our model and log its information.


-Runner object is responsible for creating mini-batches of observations, actions, values, rewards, etc to calculate the 
	general advantage estimation. We use mini-batches to ensure our GPU doesn't run out of memory while training. We can then
	return our mini-batches of observations, actions, returns(Qval) and values V(S) back to our learn function. 
	The major purpose of this runner object is to handle executing different environments in parallel. Utilizes step_model to generate experiences and train_model to train from experiences.
	We compute the gradients all at ounce using the train_model and our mini-batches of experience.
	

sonic_env.py
-Gives us 13 sonic levels and gives our environment some key properties
	-Discrete actions. Instead of having the total combination of the number of buttons as our action space (which is a lot...), we only allow for 7
	-Scale the rewards our environment gives us
	-Preprocess the frames for our model to process
	-Stack 4 preprocessed frames together
	-In this games particular case, we allow for backtracking. This lets the agent not get discouraged if it needs to revert progress.

architecture.py 
-Defines the architecture of the A2C model. Most importantly it allows our first FC layer to branch off to calculate V(S) and our action probability distribution independently. This allows our actor and critic to use a majority of the same network. 
