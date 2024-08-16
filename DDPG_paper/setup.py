# Envirement setups
NE             = 3           # Number of edge servers
NK             = 6           # Number of service types
PROBS          =\
    [0.0, 0.8, 0.0, 0.2, 0.0, 0.0] # Request probabilities (optimal policy); None value or probability vector
RHO_MAX        = 5           # Maximum density of vehicles within range of each edge
EPISODES       = 200         # The number of episodes
TIME_SLOTS     = 64          # Total number of time slots
DELTA          = 60          # The duration of each time slot (W)

# Learnig parameters
GAMMA          = 0.99        # Discont factor
ALPHA          = 0.0001      # Learning rate of Actor Network
BETA           = 0.0002      # Learnnig rage of Critic Network
TAU            = 0.001       # Soft update coefficient
N              = 128         # Size of mini-batch sample
D              = int(1e5)    # The size of experinece replay buffer
XI             = 0.005       # Noise added to actions
EPSILON        = 1           # Delay and energy trade off factor [0, 1]; 0 -> energy minimization and 1 -> delay minimization

# Computational parameters
LAMBDA         = 1e3         # Computation intensity for each task (Cycles per bit)
DK             = 20          # Data size of each task (Mb)
FV             = 5e8         # Cpu cycles of each vehicle (Cycles per second)
FE             = 1e9         # Cpu cycles of each edge server (Cycles per second)
E2E_POWER      = 1           # Transmission power between edge servers
E2C_POWER      = 2           # Transmission power between edge servers and cloud (W)
K              = 2e-18       # The computing energy efficiency parameter
ETA            = 2           # The energy consumption exponent factor

# Transmission parameters
E2E_RATE       = 10          # Transmission rate between edge servers (Mb per second)
E2C_RATE       = 0.4         # Transmission rate from edge to cloud (Mb per second)
BANDWIDTH      = 20          # Bandwidth of each edge server
minSINR        = 4           # Minimum Signal to noise ratio (dB)
maxSINR        = 5           # Maximum Signal to noise ratio (dB)
THETA          = 100         # Storage space of each service program (Mb)
SV             = 100         # Storage capacity of each vehicle (Mb)
SE             = 200         # Storage capacity of each edge server (Mb)
MIN_OFFLOADING = 0.20        # Minimum offloading
MAX_OFFLOADING = 1.00        # Maximum offloading

# Program setups
EVAL           = False       # Ignore adding Gaussian noie with the value True
LOAD           = False       # Loading DNN weithts with the value True. no plotting with the value True
  
# Simulation saving path
SIMU1_FIG_PATH     = '.plots/.simu1/'
SIMU2_FIG_PATH     = '.plots/.simu1/'
DENSITY_FIG        = '.plots/.simu2/density.png'
TASK_SIZE_FIG      = '.plots/.simu3/task_size.png'
LOAD_PATH          = '.checkpoints'