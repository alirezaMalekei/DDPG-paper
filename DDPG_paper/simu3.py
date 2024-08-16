from ddpg_tf2 import Agent
from utils import plot_curve
from env2 import Env
from setup import LOAD, TASK_SIZE_FIG, RHO_MAX, DK
from base import run

def simulation3():
     
    load_checkpoint = LOAD
    
    figure6 = TASK_SIZE_FIG
    
    simulations = ['D', 'LRU', 'R', 'F', 'H']
        
    plot_colors  = ['blue',
                    'yellow',
                    'gray', 
                    'black',
                    'red']
    
    plot_names   = ['DDPG-based Edge Caching',
                    'LRU Edge Caching',
                    'Random Edge Caching',
                    'No Edge Caching',
                    'Cloud Computing',]
        
    task_sizes = [20, 25, 30, 50, 60]
    
    ddpg_Tend            = []
    lru_Tend             = []
    rand_Tend            = []
    cc_Tend              = []
    no_caching_Tend      = []
    total_delays_list    = [ddpg_Tend, lru_Tend, rand_Tend, cc_Tend, no_caching_Tend]
    
    energy_history       = []
    total_delay_history  = []
    score_history        = []
        
    print("[.............simulation #3.............]")
    
    print("\nplots for different task sizes 20, 25, 30, 50, and 60\n")
    
    print("[.............training.............]")

    # Default envirement setups
    env = Env(density=RHO_MAX, task_size=DK)
    
    agent = Agent(input_dims=env.observation_shape, n_actions=env.action_shape)
    
    agent.set_prefix('ddpg_different_task_sizes', unicode=True)
    
    # Traning step
    # Note: first of all we're trying to train each model one time to ensure it works properly
    avg_delay_history, total_delay_history, energy_history, score_history, nums_history =\
        run(env, agent, target='D', load=False, train=True)

    for task_size in task_sizes:
        
        env = Env(task_size=task_size, density=RHO_MAX)
        
        print('\n...Task Size = %.i'%task_size, '\n')
     
        for (simulation, color, name, total_delays) in zip(simulations, plot_colors, plot_names, total_delays_list):
                                
            print("...%s..."%name)
                
            avg_delay_history, total_delay_history, energy_history, score_history, nums_history =\
                run(env, agent, target=simulation, load=True, train=False)
                        
            total_delays.append(total_delay_history[-1])
                                    
    print("\n\r ...Done! 🌟 \n\r")
            
    if not load_checkpoint:
        plot_curve(total_delays_list, figure6, plot_colors, plot_names, 
                   "Task Size", 
                   "Total delay in the period t_end", 
                    ys=task_sizes)
        