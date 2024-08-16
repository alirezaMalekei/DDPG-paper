from ddpg_tf2 import Agent
from utils import plot_curve, plot_clustered_column2
from env2 import Env
from base import run
from setup import LOAD, SIMU1_FIG_PATH, RHO_MAX, DK

def simulation1():
    
    env = None
    
    agent = None
    
    load_checkpoint = LOAD
        
    fig_path        = SIMU1_FIG_PATH
        
    simulations = ['D', 'O']
                
    plot_colors  = [
                    'blue',
                    'green',
                    'yellow',
                    'gray',
                    'red',
                    ]
    
    plot_names   = [
                    'DDPG-based Edge Caching (Delay Minimization)',
                    'DDPG-based Edge Caching (NO Partial Offloading)',
                    'LRU Edge Caching',
                    'No Edge Caching',
                    'Cloud Computing',
                    ]
    
    prefixes      = [
                     'ddpg_delay_min',
                     'ddpg_no_partial_offloading',
                     ]
    
    total_delay_data    = []
    total_delay_data2   = []
    total_delay_history = []
    avg_delay_data      = []
    avg_delay_data2     = []
    avg_delay_history   = []
    energy_data         = []
    energy_data2        = []
    energy_history      = []
    score_history       = []
    nums_history        = []
    agents              = []
    agent               = None
    env                 = Env(density=RHO_MAX, task_size=DK)
    
    print("[.............simulation #1.............]")
    
    print('analysis of different service caching strategies in terms of delays and energy consumption')
    
    print("[.............training.............]")
    
    for (simulation, prefix) in zip(simulations, prefixes):
                
        agent = Agent(input_dims=env.observation_shape, n_actions=env.action_shape)
        
        agent.set_prefix(prefix=prefix, unicode=True)
        
        agents.append(agent)
        
        # Training step
        # Note: first of all we're trying to train each model one time to ensure it works properly
        avg_delay_history ,total_delay_history, energy_history, score_history, nums_history =\
            run(env, agent, target=simulation, load=False, train=True)
            
        print('...')
                                     
    for (simulation, name, agent) in zip(simulations, plot_names[0:2], agents):
        
        print("...%s..."%name)

        avg_delay_history ,total_delay_history, energy_history, score_history, nums_history =\
            run(env, agent, target=simulation, load=True, train=False)
        
        total_delay_data.append(total_delay_history)
        
        avg_delay_data.append(avg_delay_history)
        
        energy_data.append(energy_history)
        
        total_delay_data2.append(total_delay_history[-10:])
        
        avg_delay_data2.append(avg_delay_history[-10:])
        
        energy_data2.append(energy_history[-10:])
                                                                    
    simulations = ['LRU', 'F','H']
    
    for (simulation, name) in zip(simulations, plot_names[2:5]):
        
        print("...%s..."%name)

        avg_delay_history ,total_delay_history, energy_history, score_history, nums_history =\
            run(env, None, target=simulation, load=False, train=False)
                            
        total_delay_data.append(total_delay_history)
        
        avg_delay_data.append(avg_delay_history)
        
        energy_data.append(energy_history)
        
        total_delay_data2.append(total_delay_history[-10:])
        
        avg_delay_data2.append(avg_delay_history[-10:])
        
        energy_data2.append(energy_history[-10:])
                                                           
    print("\n\r ...Done! 🌟 \n\r")   
        
    if not load_checkpoint:
        
        plot_curve(total_delay_data, fig_path + 'total_delay.png', plot_colors, plot_names, 'episode', 'total delay (Second)')
        
        plot_curve(avg_delay_data, fig_path + 'average_delay.png', plot_colors, plot_names, 'episode', 'average delay (Second)')
        
        plot_curve(energy_data, fig_path + 'total_energy.png', plot_colors, plot_names, 'episode', 'total energy (Watt)')
        
        plot_clustered_column2(plot_names, total_delay_data2, fig_path + 'total_delay2.png', 
                               'episode', 'total delay (Second)', bar_width=0.15, colors=plot_colors)

        plot_clustered_column2(plot_names, avg_delay_data2, fig_path + 'average_delay2.png', 
                               'episode', 'average delay (Second)', bar_width=0.15, colors=plot_colors)

        plot_clustered_column2(plot_names, energy_data2, fig_path + 'total_energy2.png', 
                               'episode', 'total energy of the edge servers (Watt)', bar_width=0.15, colors=plot_colors)
        