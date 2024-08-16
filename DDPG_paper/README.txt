1. delay analysis:
    
        local execution time              = 40 second per task
        vehicle to edge transmission time = 2  second per task
        edge execution time               = 20 second per task
        edge to edge transmission time    = 2  second per task
        edge to cloud transmission time   = 50 second per task
        
        minimum task execution time       = 2 + 20 = 22 second by edge execution
    
        maximum task execution time       = 2 + 50 = 52 second by cloud execution

2. edge energy consumption analysis:

        edge execution             = 40  W per task
        edge to edge transmission  = 2   W per task
        edge to cloud transmission = 100 W per task
        
        minimum energy consumption of edge servers for each task = 0      by vehicle execution
        
        maximum energy consumption of edge servers for each task = 100 W  uploading to cloud
    
3. setup tunes:

        # Envirement setups
        NE              = 3         # Number of edge servers
        NK              = 6         # Number of service types
        PROBS           =\
                [0.0, 0.8, 0.0, 0.2, 0.0, 0.0] # Request probabilities (optimal policy); None value or probability vector
        RHO_MAX        = 5          # Maximum density of vehicles within range of each edge
        EPISODES       = 200        # The number of episodes
        TIME_SLOTS     = 64         # Total number of time slots
        DELTA          = 60         # The duration of each time slot (W)
        
        # Learnig parameters
        GAMMA          = 0.99       # Discont factor
        ALPHA          = 0.0001     # Learning rate of Actor Network
        BETA           = 0.0002     # Learnnig rage of Critic Network
        TAU            = 0.001      # Soft update coefficient
        N              = 128        # Size of mini-batch sample
        D              = 10000      # The size of experinece replay buffer
        XI             = 0.005      # Noise added to actions
        EPSILON        = 1          # Delay and energy trade off factor [0, 1]; 0 -> energy minimization and 1 -> delay minimization
        
        # Computational parameters
        LAMBDA         = 1e3        # Computation intensity for each task (Cycles per bit)
        DK             = 20         # Data size of each task (Mb)
        FV             = 5e8        # Cpu cycles of each vehicle (Cycles per second)
        FE             = 1e9        # Cpu cycles of each edge server (Cycles per second)
        E2E_POWER      = 1          # Transmission power between edge servers
        E2C_POWER      = 2          # Transmission power between edge servers and cloud (W)
        K              = 2e-18      # The computing energy efficiency parameter
        ETA            = 2          # The energy consumption exponent factor
        
        # Transmission parameters
        E2E_RATE       = 10         # Transmission rate between edge servers (Mb per second)
        E2C_RATE       = 0.4        # Transmission rate from edge to cloud (Mb per second)
        BANDWIDTH      = 20         # Bandwidth of each edge server
        minSINR        = 4          # Minimum Signal to noise ratio (dB)
        maxSINR        = 5          # Maximum Signal to noise ratio (dB)
        THETA          = 100        # Storage space of each service program (Mb)
        SV             = 100        # Storage capacity of each vehicle (Mb)
        SE             = 200        # Storage capacity of each edge server (Mb)
        MIN_OFFLOADING = 0.20       # Minimum offloading
        MAX_OFFLOADING = 1.00       # Maximum offloading

4. Implementation Article https://ieeexplore.ieee.org/abstract/document/10007043 

        @article{xue2023joint,
          title={Joint service caching and computation offloading scheme 
                  based on deep reinforcement learning in vehicular edge computing systems},
          author={Xue, Zheng and Liu, Chang and Liao, Canliang and Han, Guojun and Sheng, Zhengguo},
          journal={IEEE Transactions on Vehicular Technology},
          volume={72},
          number={5},
          pages={6709--6722},
          year={2023},
          publisher={IEEE}
        }
    
    the Authors proposed an edge caching and offloading scheme based on deep deterministic policy gradient (DDPG) to efficiently make decisions on task offloading and service caching.

4. you can concisely go deep into DDPG by clicking on link https://spinningup.openai.com/en/latest/algorithms/ddpg.html. but, I wrote a brief letters below.

    Deep Deterministic Policy Gradient (DDPG) is a deep reinforcement learning algorithm designed for continuous action spaces.
    It combines elements from both Q-learning and policy gradient methods to learn a policy and a Q-function concurrently. 
    Here are some key points about DDPG:
    
    Key Concepts
        1. Q-Learning Side:
            DDPG uses the Bellman equation to learn the Q-function, which estimates the expected return (reward) of taking a certain action in a given state and following the policy thereafter.
            The Q-function is updated using off-policy data, meaning it can learn from actions that were not taken by the current policy.
        2. Policy Learning Side:
            The policy in DDPG is deterministic, meaning it outputs a specific action for each state rather than a probability distribution over actions.
            The policy is updated using the Q-function, which guides the policy towards actions that maximize the expected return.
        3. Continuous Action Spaces:
            Unlike traditional Q-learning, which works well with discrete action spaces, DDPG is specifically designed for environments where actions are continuous.
            This is achieved by using a deterministic policy that can output continuous actions directly.
        4. Replay Buffer and Target Networks:
            DDPG uses a replay buffer to store past experiences (state, action, reward, next state) and sample them randomly during training. 
            This helps break the correlation between consecutive experiences and stabilizes training.
            Target networks are used to stabilize the learning process by providing a slowly updating reference for the Q-function and policy updates.
    
5. good luck
        