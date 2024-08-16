import numpy as np
from setup import EPISODES, TIME_SLOTS, EVAL
from env2 import Env
# from progress import bar

def run(env, agent, target='D', load=False, train=True):
        
    episodes              = EPISODES
    maxTimeSlot           = TIME_SLOTS
    best_score            = float("-inf")
    energy_history        = []
    score_history         = []
    nums_history          = []
    total_delay_history   = []
    avg_delay_histroy     = []
    
    load_checkpoint       = load
    action                = None
    flatAct               = None
    obs                   = None
    init_obs              = None
    flatObs               = None
    flatObs_              = None
    reward                = float("-inf")
    done                  = False
    info                  = None
    total_delay           = 0
    total_energy          = 0
    score                 = 0
    nums                  = []
    cond                  = target == 'D' or target =='E' or target =='O'
    density               = env.max_vehicle_density
    task_size             = env.task_size
    
    not_converged_after_itrs   = 200
    
    if load_checkpoint:
        for _ in range(agent.batch_size):
            obs, info = env.reset()
            flatObs = env.flattenObservation(obs)
            
            action = env.action_space.sample()
            flatAct = env.flattenAction(action)
            
            flatObs_, flatObs, reward ,done, info, avg_delay, total_delay, total_energy, nums =\
                env.step(action, flat=False, target=target)
                
            if cond:   
                agent.remember(flatObs, flatAct, reward, flatObs_, done)
                
        if cond:    
            agent.learn()
            agent.load_models()
        evaluate = not EVAL
    else:
        # Adding a random Gaussian noise
        evaluate = EVAL
        
    itr = 0
    
    while True:
        
        if not train and itr > episodes:
            break
        
        itr += 1
        
        init_obs, info = env.reset()
        obs = init_obs
        done = False
        score = 0
        
        flatObs = env.flattenObservation(obs)
        
        if cond:
            flatAct = agent.choose_action(flatObs, evaluate)
        else:
            # select a random action
            action = Env(density=density, task_size=task_size).action_space.sample()
            flatAct = env.flattenAction(action)
        
        flatObs_, flatObs, reward ,done, info, avg_delay, total_delay, total_energy, nums =\
            env.step(flatAct, flat=True, target=target)
            
        if cond:
            agent.remember(flatObs, flatAct, reward, flatObs_, done)
        
        # Saving histories
        total_delay_history.append(total_delay)
        avg_delay_histroy.append(avg_delay)
        energy_history.append(total_energy)
        nums_history.append(nums)
        
        # progress = (itr + 1) / episodes
        # bar_length = 35
        # filled_length = int(bar_length * progress)
        # label = "...processing..."
        # bar = f"[{'#' * filled_length}{'-' * (bar_length - filled_length)}] {progress * 100:.1f}%"
        # print(f"\r {label} {bar}", end="")
        
        print('episode = %i'%itr, 
              '; total delay = %.1f'% total_delay,
              '; avg delay = %.1f'% avg_delay,
              '; total energy = %.1f'% total_energy,
              '; reward = %.1f'% reward, 
              '; n_local = %.1f'% nums[0],
              '; n_partial = %1.f'% float(nums[5] + nums[6]),
              '; n_edge = %.1f'% nums[1],
              '; n_pool = %.1f'% nums[2],
              '; n_cloud = %.1f'% nums[3],
              '; task no = %1.f'% nums[4],)
            
        if done and train:
            print("\n...the algorithm converged")
            break
        
        if not done and train and itr > not_converged_after_itrs:
            itr = 0
            print('...reinitializing dnn weights')
            agent.set_weights()
            env.close()

        for timeSlot in range(maxTimeSlot):
              
            if not cond or not train:
                break
            
            flatAct = agent.choose_action(flatObs, evaluate)
            
            flatObs_, flatObs, reward ,done, info, avg_delay, total_delay, total_energy, nums =\
                env.step(flatAct, flat=True, target=target)
            
            score += reward
                               
            agent.remember(flatObs, flatAct, reward, flatObs_, done)
            
            agent.learn()
                
            flatObs = flatObs_
            
        if cond and train:
            score_history.append(score)
            avg_score = np.mean(score_history[-1:])
            if avg_score > best_score:
                best_score = score
                agent.save_models()
                
    env.close()
                
    return avg_delay_histroy, total_delay_history, energy_history, score_history, nums_history
