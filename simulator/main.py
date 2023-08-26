# import rl simulator and visualizations from pyfit
# set parameters from reward_func, state_evolution, model, and stuff from config
# in config, we have things like the state size, initial state, action space size, and hyperparameters from the simulation 
# hyperparameters can be things like num_rounds, 

# set up the simulator object 
# call simulator.step. Can access class things like reward_list, current_state, action_list, 
# maybe we can have a visualizer function where you can pass in a mask for columns which are the same, and also names, for example things that are 2 transfers, and then be able to plot things from there

# by the end, you will be able to plot rewards, and you can get meta information from the action list. 
# instead, you can call it many times, and it will record only the last lists, and maintain a throughout simulation histogram 
# you can also import compare models class, which will run this many times, given different model imports, 
    # and give you access to good plotting functions 

# lastly, we need to put something in constants which is the root path to this dir, so we can properly import in this subdir 