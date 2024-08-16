from ddpg_tf2 import Agent
from utils import plot_clustered_column
from env2 import Env
from base import run
from setup import LOAD, DENSITY_FIG, DK

def simulation2():
     env = None
     
     agent = None
     
     load_checkpoint = LOAD
     
     figure5 = DENSITY_FIG
     
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
     
     densities     = [2, 4, 5, 10]
     
     values              = []
     energy_history      = []
     total_delay_history = []
     score_history       = []
     env                 = None
     agent               = None
     
     print("[.............simulation #2.............]")
     
     print("\ndelays for different vehicle densities 2, 4, 5, and 10\n")
     
     for density in densities:
         
         print("[.............training.............]")
         
         env = Env(density=density, task_size=DK)
         
         agent = Agent(input_dims=env.observation_shape, n_actions=env.action_shape)
         
         agent.set_prefix('ddpg_different_densites', unicode=True)
         
         # Traning step
         # Note: first of all we're trying to train each model one time to ensure it works properly
         avg_delay_history ,total_delay_history, energy_history, score_history, nums_history =\
             run(env, agent, target='D', load=False, train=True)
         
         print('\n...Density = %.i'%density, '\n')
               
         for (simulation, color, name) in zip(simulations, plot_colors, plot_names):
                  
            print("...%s..."%name)
                
            avg_delay_history ,total_delay_history, energy_history, score_history, nums_history =\
                run(env, agent, target=simulation,  load=True, train=False)
                        
            values.append(total_delay_history[-1])
                                                    
     print("\n\r ...Done! 🌟 \n\r")
        
     if not load_checkpoint:
         plot_clustered_column(densities, plot_names, values, figure5, 
                               'The vehicle density at each edge', 
                               'Total delay in period t end', 
                                bar_width=0.15)
         