import gymnasium as gym
import numpy as np
import setup
from gymnasium.spaces import Box, Tuple
from tensorflow import clip_by_value

class Env(gym.Env):
    def __init__(self, task_size=20, density=2):
        super(Env, self).__init__()

        self.edge_no                = setup.NE
        self.max_service_program    = setup.NK
        self.max_vehicle_density    = density
        self.channel_bandwidth      = setup.BANDWIDTH
        self.minSINR                = setup.minSINR
        self.maxSINR                = setup.maxSINR
        self.max_time_slot          = setup.TIME_SLOTS
        self.request_probs          = setup.PROBS
        # Min partial offloadig
        self.min_offloading         = setup.MIN_OFFLOADING
        # Max partial offloading
        self.max_offloading         = setup.MAX_OFFLOADING
        # Compromise factor -> a trade off for delay and energy minimization
        self.epsilon                = setup.EPSILON
        self.info                   = {}
        self.vehicle_max_caching    = np.floor(setup.SV / setup.THETA)
        self.edge_max_caching       = np.floor(setup.SE / setup.THETA)
        # Transmission time between vehicles and edge nodes calculated in every time slot
        self.V2E_trans_time_        = None
        # Energy Efficiency parameter
        self.enerty_cons_factor     = setup.K * np.power(setup.FE, setup.ETA)
        # Transmission power between edge servers
        self.edge_power             = setup.E2E_POWER
        # Transmission power between edge servers and cloud (W)
        self.cloud_power            = setup.E2C_POWER

        # LRU Matrix for edge caching
        self.lruMatrix =\
            np.zeros((self.max_vehicle_density *
                     self.edge_no, self.max_service_program))
        
        # ...
        self.task_size = task_size

        # ...
        self.vehicle_exec_time = setup.LAMBDA * \
            (self.task_size * 1e6 / setup.FV)
        # ...
        self.edge_exec_time = setup.LAMBDA * (self.task_size * 1e6 / setup.FE)

        # Transmission time between edge servers
        self.E2E_trans_time = self.task_size / setup.E2E_RATE
        # Transmission time between each edge server and the cloud
        self.E2C_trans_time = self.task_size / setup.E2C_RATE

        # Max tolerable delay (Constant)
        allocated_ch_bw = self.channel_bandwidth / self.max_vehicle_density
        max_V2E_trans_time = self.task_size / \
            ((np.log2(1 + self.minSINR) * allocated_ch_bw))
        TDMAX1 = max_V2E_trans_time + self.edge_exec_time
        TDMAX2 = max_V2E_trans_time + self.E2C_trans_time
        self.max_delay = max(self.vehicle_exec_time,  TDMAX1, TDMAX2)

        # Min delay (constant)
        min_V2E_trans_time = self.task_size / \
            ((np.log2(1 + self.maxSINR) * allocated_ch_bw))
        TDMIN1 = min_V2E_trans_time + self.edge_exec_time
        TDMIN2 = min_V2E_trans_time + self.E2C_trans_time
        self.min_delay = min(self.vehicle_exec_time, TDMIN1, TDMIN2)
        
        # Convergence delay
        max_edge_exec_time = TDMAX1 + self.E2E_trans_time
        max_cloud_exec_time = max_V2E_trans_time + self.E2C_trans_time
        self.convergence_delay = min(self.vehicle_exec_time, 
                                     max_edge_exec_time, max_cloud_exec_time)
        # Convergence counter
        self.conv_cntr = 10 * setup.N 

        # Max energy consumption by each task (constant)
        ECMAX1 = self.edge_power * self.E2E_trans_time + \
            self.enerty_cons_factor * self.edge_exec_time
        ECMAX2 = self.cloud_power * self.E2C_trans_time
        self.max_energy_consu = max(ECMAX1, ECMAX2)

        # Min energy consumption of edge servers when receive a task (constant)
        ECMIN1 = self.enerty_cons_factor * self.edge_exec_time
        ECMIN2 = self.cloud_power * self.E2C_trans_time
        self.min_energy_consu = min(ECMIN1, ECMIN2)

        # flatten action shape
        self.action_shape =\
            np.array([self.edge_no * self.max_service_program *
                     (self.max_vehicle_density + 1 + 1)])

        # flatten observation shape
        self.observation_shape =\
            np.array([self.edge_no * (self.max_service_program * (2 * self.max_vehicle_density + 1) 
                                      + 2 * self.max_vehicle_density)])

        # action space
        self.edgeServicCachingIndicators = []
        self.vehicleOffloadingPreportions = []
        self.edgeOffloadingPreportions = []

        edgeServiceCachingIndicator = None
        high = None

        for _ in range(self.edge_no):
            # Constrainte on max storage capacity of the edges
            edgeServiceCachingIndicator = Box(low=0, high=1, 
                                              shape=(1, self.max_service_program), dtype=np.bool_)
            high = self.restrict(restriction=self.edge_max_caching)
            edgeServiceCachingIndicator.high = high
            edgeServiceCachingIndicator.low = high
            self.edgeServicCachingIndicators.append(edgeServiceCachingIndicator)

            # ...
            self.vehicleOffloadingPreportions.append(Box(low=self.min_offloading, high=self.max_offloading, 
                                                         shape=(self.max_vehicle_density, 
                                                                self.max_service_program), dtype=np.float32))
            # ...
            self.edgeOffloadingPreportions.append(Box(low=self.min_offloading, high=self.max_offloading, 
                                                      shape=(1, self.max_service_program), 
                                                      dtype=np.float32))

        self.action_space = Tuple((
            Tuple((self.edgeServicCachingIndicators[i])
                  for i in range(self.edge_no)),
            Tuple((self.vehicleOffloadingPreportions[i])
                  for i in range(self.edge_no)),
            Tuple((self.edgeOffloadingPreportions[i]) 
                  for i in range(self.edge_no))))

        # observation space
        self.vehicleRequestIndicators = []
        self.vehicleServiceCachingIndicators = []
        self.allocatedBandwidths = []
        self.recievedSINRs = []
        vehicleRequestIndicator = []
        vehicleServiceCachingIndicator = None

        for _ in range(self.edge_no):
            # each vehicle can only send one task request in time slot t
            high = self.restrict(restriction=self.vehicle_max_caching, probabilities=self.request_probs)
            for i in range(self.max_vehicle_density-1):
                high = np.vstack((high, 
                                  self.restrict(restriction=self.vehicle_max_caching, 
                                                probabilities=self.request_probs)))

            vehicleRequestIndicator = Box(low=0, high=1, 
                                          shape=(self.max_vehicle_density, self.max_service_program), dtype=np.bool_)
            vehicleRequestIndicator.high = high
            # each vehicle launch at least one request in every time slot t
            vehicleRequestIndicator.low = high
            self.vehicleRequestIndicators.append(vehicleRequestIndicator)

            # ...
            vehicleServiceCachingIndicator = Box(low=0, high=1, 
                                                 shape=(self.max_vehicle_density, self.max_service_program), dtype=np.bool_)
           # Constrainte on max storage capacity of the vehicles
            high = self.restrict(restriction=self.vehicle_max_caching)
            for i in range(self.max_vehicle_density-1):
                high = np.vstack((high, self.restrict(restriction=self.vehicle_max_caching)))
            vehicleServiceCachingIndicator.high = high
            vehicleServiceCachingIndicator.low = high
            self.vehicleServiceCachingIndicators.append(vehicleServiceCachingIndicator)

            # ...
            allocatedBandwith = self.channel_bandwidth / self.max_vehicle_density
            self.allocatedBandwidths.append(Box(low=allocatedBandwith, high=allocatedBandwith, 
                                                shape=(self.max_vehicle_density, 1)))
            # ...
            self.recievedSINRs.append(Box(low=self.minSINR, high=self.maxSINR, 
                                          shape=(1, self.max_vehicle_density), dtype=np.float32))

        self.observation_space = Tuple((
            Tuple((self.vehicleRequestIndicators[i])
                  for i in range(self.edge_no)),
            Tuple((self.vehicleServiceCachingIndicators[i]) 
                  for i in range(self.edge_no)),
            Tuple((self.allocatedBandwidths[i]) 
                  for i in range(self.edge_no)),
            Tuple((self.recievedSINRs[i]) 
                  for i in range(self.edge_no)),
            Tuple((self.edgeServicCachingIndicators[i])     
                  for i in range(self.edge_no))))

        # Make envirement
        self.observation = self.newObservation()
        self.observation_ = None

    def restrict(self, restriction, probabilities=None):
        high = np.random.randint(2, size=(1, self.max_service_program))
        if probabilities is not None:
            prob_sum = np.sum(probabilities)
            cond = (len(probabilities) ==
                    self.max_service_program) and prob_sum == 1
            if not cond:
                probabilities = None
                self.restrict(restriction=restriction, probabilities=None)
            else:
                services = np.arange(self.max_service_program)
                sample = np.random.choice(
                    services, size=self.max_service_program, p=probabilities)
                indices = []
                [indices.append(x) for x in sample if x not in indices]
                high = np.zeros_like(services)
                high[indices] = 1

        ones = high[high > 0]
        if ones.size > restriction:
            size = int(ones.size - restriction)
            random_indices = np.random.choice(ones.size, size=size, replace=False)
            random_flipped = ones
            random_flipped[random_indices] = 0
            high[high > 0] = random_flipped
            
        elif ones.size == 0:
            random_index = np.random.choice(self.max_service_program, size=1, replace=False)
            high[0][random_index] = 1
        else:
            pass
            
        high[high > 0] = 1

        return high

    def reset(self):
        # Make envirement
        self.observation = self.newObservation()
        self.observation_ = None

        self. info = {}

        return self.observation, self.info

    def step(self, action=None, flat=False ,target='D'):
        """
        D stands for Delay Minimization
        E stands for Energy Minimization
        C stands for without any edge caching
        O stands for no partial offloading
        LRU stanss for Last Recently Used edge caching 
        R stands for random edge caching
        H stands for executing all tasks on cloud, 
            no task execute neither on vehicles nor on edge servers.
        """
        if self.observation_ is not None:
            self.observation = self.observation_

        # Edge Caching And Offloding
        if target == 'D':
            self.epsilon = 1
            observation_, observation, avg_delay, total_delay, total_energy, reward, done, info, nums =\
                self.partialOffloading(action=action, flat=flat)

        elif target == 'E':
            self.epsilon = 0
            observation_, observation, avg_delay, total_delay, total_energy, reward, done, info, nums =\
                self.partialOffloading(action=action, flat=flat)

        elif target == 'F':
            self.epsilon = 1
            observation_, observation, avg_delay, total_delay, total_energy, reward, done, info, nums =\
                self.noEdgeCaching(action=action, flat=flat)

        elif target == 'O':
            self.epsilon = 1
            observation_, observation, avg_delay, total_delay, total_energy, reward, done, info, nums =\
                self.noPartialOffloading(action=action, flat=flat)

        elif target == 'R':
            self.epsilon = 1
            observation_, observation, avg_delay, total_delay, total_energy, reward, done, info, nums =\
                self.randomEdgeCaching(action=action, flat=flat)

        elif target == 'H':
            self.epsilon = 1
            observation_, observation, avg_delay, total_delay, total_energy, reward, done, info, nums =\
                self.offloadingToCloud(action=action, flat=flat)

        elif target == 'LRU':
            self.epsilon = 1
            observation_, observation, avg_delay, total_delay, total_energy, reward, done, info, nums =\
                self.LRUedgeCaching(action=action, flat=flat)

        else:
            print('Just D, E, C and O are valid input keys')
            return

        return observation_, observation, reward, done, info, avg_delay, total_delay, total_energy, nums

    def LRUedgeCaching(self,  action, flat=False):

        I, CV, B, X, CE = self.observationSampleDecode(self.observation)

        self.lruMatrix = I + self.lruMatrix
        replacement    = np.sum(self.lruMatrix, axis=0)

        # Constraint on storage capacity of edge caching
        while True:
            ones = replacement[replacement > 0]
            if ones.size <= self.edge_max_caching:
                break
            ones[np.argmin(ones)] = 0
            replacement[replacement > 0] = ones

        replacement[replacement > 0] = 1
        
        cachingPolicy = replacement
        
        for _ in range(self.edge_no - 1):
            cachingPolicy = np.vstack((cachingPolicy, replacement))
            
        return self.partialOffloading(action=action, flat=flat, policy=cachingPolicy)

    def randomEdgeCaching(self,  action, flat=False):
        return self.partialOffloading(action=action, flat=flat)

    def offloadingToCloud(self,  action, flat=False):

        I, CV, B, X, CE = self.observationSampleDecode(self.observation)

        # Maximum Vehicle to edge rate is log2(1 + 5) * 20 == 51.69925001442312
        # Minimum Vehicle to edge rate is log2(1 + 4) * 4 == 9.287712379549449 or
        # Minimum Vehicle to edge rate is log2(1 + 5) * 4 == 10.339850002884624
        # Transmissin rate between vehicles and the nearest edge nodes
        V2XRate = np.log2(1 + X) * B
        V2XRate[V2XRate == 0] = -1

        # Transmission time between vehicles and the nearest edge nodes
        self.V2E_trans_time_ = (setup.DK / V2XRate)
        self.V2E_trans_time_[self.V2E_trans_time_ < 0] = 0

        local_exec_time = 0
        edge_exec_time  = 0
        pool_exec_time  = 0
        cloud_exec_time = 0
        V2ETransTime    = 0
        E2ETransTime    = 0
        E2CTransTime    = 0
        task_no = np.sum(I)
        
        n_local = 0
        n_edge  = 0
        n_pool  = 0
        n_cloud = 0
        
        n_v2e_partial = 0
        n_e2e_partial = 0  
        
        sum_delay = 0

        # All tasks execues on cloud
        cloud = I
        V2ETransTime += np.sum(cloud * self.V2E_trans_time_)
        E2CTransTime += np.sum(cloud * self.E2C_trans_time)
        cloud_exec_time += V2ETransTime + E2CTransTime
        n_cloud += np.sum(cloud)
        
        nums = [n_local, n_edge, n_pool, n_cloud, task_no, n_v2e_partial, n_e2e_partial]
        
        sum_delay += local_exec_time + edge_exec_time + pool_exec_time + cloud_exec_time
         
        execTimes = [local_exec_time, edge_exec_time, pool_exec_time, cloud_exec_time, 
                     0, 0, sum_delay]
         
        transTimes = [E2ETransTime, E2CTransTime]
        
        deObs = [I, CV, B, X, CE]

        return self.func(execTimes, nums, transTimes, deObs)

    def noEdgeCaching(self, action, flat=False):
        if flat == False:
            CE_, OV, OE = self.actionSampleDecode(action)
        else:
            CE_, OV, OE = self.extractAction(action)

        I, CV, B, X, CE = self.observationSampleDecode(self.observation)

        # Maximum Vehicle to edge rate is log2(1 + 5) * 20 == 51.69925001442312
        # Minimum Vehicle to edge rate is log2(1 + 4) * 4 == 9.287712379549449 or
        # Minimum Vehicle to edge rate is log2(1 + 5) * 4 == 10.339850002884624
        # Transmissin rate between vehicles and the nearest edge nodes
        V2XRate = np.log2(1 + X) * B
        V2XRate[V2XRate == 0] = -1

        # Transmission time between vehicles and the nearest edge nodes
        self.V2E_trans_time_ = (setup.DK / V2XRate)
        self.V2E_trans_time_[self.V2E_trans_time_ < 0] = 0

        local_exec_time = 0
        edge_exec_time  = 0
        pool_exec_time  = 0
        cloud_exec_time = 0
        V2ETransTime    = 0
        E2ETransTime    = 0
        E2CTransTime    = 0
        task_no = np.sum(I)

        n_local = 0
        n_edge  = 0
        n_pool  = 0
        n_cloud = 0

        n_v2e_partial = 0
        n_e2e_partial = 0
        
        sum_delay = 0
        
        local = (I * CV)
        local_exec_time += np.sum(local * self.vehicle_exec_time)
        n_local += np.sum(local)

        # Request not cached on the vehicles will be offloading compeletely to cloud
        fullyOffloading = I * (1 - CV)

        V2ETransTime += np.sum(fullyOffloading * self.V2E_trans_time_)
        E2CTransTime += np.sum(fullyOffloading * self.E2C_trans_time)
        cloud_exec_time += V2ETransTime + E2CTransTime
        n_cloud += np.sum(fullyOffloading)

        nums = [n_local, n_edge, n_pool, n_cloud, task_no, n_v2e_partial, n_e2e_partial]
        
        sum_delay += local_exec_time + edge_exec_time + pool_exec_time + cloud_exec_time
         
        execTimes = [local_exec_time, edge_exec_time, pool_exec_time, cloud_exec_time, 
                     0, 0, sum_delay]
         
        transTimes = [E2ETransTime, E2CTransTime]
        
        deObs = [I, CV, B, X, CE_]
        
        return self.func(execTimes, nums, transTimes, deObs)

    def partialOffloading(self, action, flat=False, policy=None):

        if flat == False:
            CE_, OV, OE = self.actionSampleDecode(action)
        else:
            CE_, OV, OE = self.extractAction(action)

        I, CV, B, X, CE = self.observationSampleDecode(self.observation)

        if policy is not None:
            CE_ = policy

        # Maximum Vehicle to edge rate is log2(1 + 5) * 20 == 51.69925001442312
        # Minimum Vehicle to edge rate is log2(1 + 4) * 4 == 9.287712379549449 or
        # Minimum Vehicle to edge rate is log2(1 + 5) * 4 == 10.339850002884624
        # Transmissin rate between vehicles and the nearest edge nodes
        V2XRate = np.log2(1 + X) * B
        V2XRate[V2XRate == 0] = -1

        # Transmission time between vehicles and the nearest edge nodes
        self.V2E_trans_time_ = (setup.DK / V2XRate)
        self.V2E_trans_time_[self.V2E_trans_time_ < 0] = 0

        local_exec_time = 0
        edge_exec_time  = 0
        pool_exec_time  = 0
        cloud_exec_time = 0
        E2ETransTime    = 0
        E2CTransTime    = 0
        task_no = np.sum(I)

        local                = None
        partialOffloading    = None
        poolCachingIndicator = None
        poolCache            = None

        edgeCachingIndicators = []
        cachedOnEdges         = []
        edgeOffloadings       = []
        
        maxVehiclePartialOffloadingExecTime = []
        maxEdgePartialOffloadingExecTime    = []

        n_local   = 0
        n_edge    = 0
        n_pool    = 0
        n_cloud   = 0
        n_v2e_partial = 0
        n_e2e_partial = 0
        
        sum_delay = 0
        
        f = 0
        l = self.max_vehicle_density

        for i in range(self.edge_no):
            f = i + 1

            # Cached on vehicles and not Cached on the nearest edges
            local = (((I[i*l:f*l, :] * CV[i*l:f*l, :])) * (1 - CE_[i:f, :]))
            local_exec_time += np.sum(local * self.vehicle_exec_time)
            n_local += np.sum(local)

            # Cached on vehicles and cached on the nearest edges
            partial_local     = (((I[i*l:f*l, :] * CV[i*l:f*l, :])) * CE_[i:f, :]) * (1 - OV[i*l:f*l, :])
            partialOffloading = (((I[i*l:f*l, :] * CV[i*l:f*l, :])) * CE_[i:f, :]) * OV[i*l:f*l, :]
            
            v2e_partial_exec = partial_local * self.vehicle_exec_time
            
            sum_delay += np.sum(v2e_partial_exec)
            
            maxVehiclePartialOffloadingExecTime.append(np.max(v2e_partial_exec))
                        
            n_v2e_partial += np.sum(partial_local)

            v2e_partial_exec =\
                (partialOffloading * self.V2E_trans_time_[i*l:f*l, :])\
                    + (partialOffloading * self.edge_exec_time)
            
            sum_delay += np.sum(v2e_partial_exec)
            
            maxVehiclePartialOffloadingExecTime.append(np.max(v2e_partial_exec))
                        
            n_v2e_partial += np.sum(partialOffloading)

            edgeCachingIndicators.append(CE_[i:f, :])
 
            # Request not cached on the vehicles will be offloading compeletely
            fullyOffloading = I[i*l:f*l, :] * (1 - CV[i*l:f*l, :])

            # Not cached on vehicles but cached on the nearest edge
            cachedOnEdge = fullyOffloading * (CE_[i:f, :])
            cachedOnEdges.append(cachedOnEdge)

            # Neither cached on the vehicles nor the nearest edges
            edgeOffloading = fullyOffloading * (1 - CE_[i:f, :])
            edgeOffloadings.append(edgeOffloading)

        for i in range(self.edge_no):
            f = i + 1
            # Pool cache indicator
            if self.edge_no > 1:
                poolCachingIndicator = edgeCachingIndicators.copy()
                poolCachingIndicator.pop(i)
                poolCache = poolCachingIndicator.pop(0)
                for Indicator in poolCachingIndicator:
                    poolCache = np.logical_or(poolCache, Indicator)
            else:
                poolCache = 0

            # Execution on the edge completely
            edge = cachedOnEdges[i] * (1 - poolCache)
            edge_exec_time += np.sum(edge * self.V2E_trans_time_[i*l:f*l, :])
            edge_exec_time += np.sum(edge * self.edge_exec_time)
            n_edge += np.sum(edge)

            # Partial Offloading from edge to pool
            partial_edge = cachedOnEdges[i] * poolCache * (1 - OE[i:f, :])
            partialOffloading = cachedOnEdges[i] * poolCache * OE[i:f, :]
            
            
            e2e_partial_exec =\
                (partial_edge * self.V2E_trans_time_[i*l:f*l, :])\
                    + (partial_edge * self.edge_exec_time) 
                    
            sum_delay += np.sum(e2e_partial_exec)
            
            maxEdgePartialOffloadingExecTime.append(np.max(e2e_partial_exec))
                        
            n_e2e_partial += np.sum(partial_edge)

            # for energy consumption
            E2ETransTime += np.sum(partialOffloading * self.E2E_trans_time)
            
            e2e_partial_exec =\
                (partialOffloading * self.V2E_trans_time_[i*l:f*l, :])\
                    + (partialOffloading * self.E2E_trans_time)\
                    + (partialOffloading * self.edge_exec_time)
                    
            sum_delay += np.sum(e2e_partial_exec)

            maxEdgePartialOffloadingExecTime.append(np.max(e2e_partial_exec))
                        
            n_e2e_partial += np.sum(partialOffloading)

            # Service programs of some requests launched by moving vehicles
            # are not cached neither on the nearest edge nor on the vehicles
            pool = edgeOffloadings[i] * poolCache

            # for energy consumption
            E2ETransTime += np.sum(pool * self.E2E_trans_time)

            pool_exec_time += np.sum(pool * self.V2E_trans_time_[i*l:f*l, :])
            pool_exec_time += np.sum(pool * self.E2E_trans_time)
            pool_exec_time += np.sum(pool * self.edge_exec_time)
            n_pool += np.sum(pool)

            # Some requests launched by vehicles transmit to cloud
            # because corresponding service programs are not cached on edge servers
            cloud = edgeOffloadings[i] * (1 - poolCache)

            # for energy consumption
            E2CTransTime += np.sum(cloud * self.E2C_trans_time)

            # for delay
            cloud_exec_time += np.sum(cloud * self.V2E_trans_time_[i*l:f*l, :])
            cloud_exec_time += np.sum(cloud * self.E2C_trans_time)
            n_cloud += np.sum(cloud)

        nums = [n_local, n_edge, n_pool, n_cloud, task_no, n_v2e_partial, n_e2e_partial]
        
        vehicle_partial_exec_time = np.max(maxVehiclePartialOffloadingExecTime)
        
        edge_partial_exec_time    = np.max(maxEdgePartialOffloadingExecTime)
        
        sum_delay += (local_exec_time + edge_exec_time + pool_exec_time + cloud_exec_time)
                 
        execTimes = [local_exec_time, edge_exec_time, pool_exec_time, cloud_exec_time, 
                     vehicle_partial_exec_time, edge_partial_exec_time, sum_delay]
          
        transTimes = [E2ETransTime, E2CTransTime]
        
        deObs = [I, CV, B, X, CE_]

        return self.func(execTimes, nums, transTimes, deObs)

    def noPartialOffloading(self,  action, flat=False):
        if flat == False:
            CE_, OV, OE = self.actionSampleDecode(action)

        else:
            CE_, OV, OE = self.extractAction(action)

        I, CV, B, X, CE = self.observationSampleDecode(self.observation)

        # Maximum Vehicle to edge rate is log2(1 + 5) * 20 == 51.69925001442312
        # Minimum Vehicle to edge rate is log2(1 + 4) * 4 == 9.287712379549449 or
        # Minimum Vehicle to edge rate is log2(1 + 5) * 4 == 10.339850002884624
        # Transmissin rate between vehicles and the nearest edge nodes
        V2XRate = np.log2(1 + X) * B
        V2XRate[V2XRate == 0] = -1

        # Transmission time between vehicles and the nearest edge nodes
        self.V2E_trans_time_ = (setup.DK / V2XRate)
        self.V2E_trans_time_[self.V2E_trans_time_ < 0] = 0

        local_exec_time = 0
        edge_exec_time  = 0
        pool_exec_time  = 0
        cloud_exec_time = 0
        E2ETransTime    = 0
        E2CTransTime    = 0
        task_no = np.sum(I)

        local = None
        poolCachingIndicator = None
        poolCache = None

        edgeCachingIndicators = []
        cachedOnEdges         = []
        edgeOffloadings       = []

        n_local = 0
        n_edge  = 0
        n_pool  = 0
        n_cloud = 0
        
        n_v2e_partial = 0
        n_e2e_partial = 0
        
        sum_delay = 0

        f = 0
        l = self.max_vehicle_density

        for i in range(self.edge_no):
            f = i + 1

            # Cached on vehicles -> locally execution
            local = (I[i*l:f*l, :] * CV[i*l:f*l, :] * (1 - CE_[i:f, :]))
            local_exec_time += np.sum(local * self.vehicle_exec_time)
            n_local += np.sum(local)
            
            edgeCachingIndicators.append(CE_[i:f, :])

            # Request not cached on the vehicles will be offloading compeletely
            fullyOffloading =\
                (I[i*l:f*l, :] * (1 - CV[i*l:f*l, :])) \
                + (I[i*l:f*l, :] * CV[i*l:f*l, :] * CE_[i:f, :])
            
            # Not cached on vehicles but cached on the nearest edge
            cachedOnEdge = fullyOffloading * (CE_[i:f, :])
            cachedOnEdges.append(cachedOnEdge)

            # Neither cached on the vehicles nor the nearest edges
            edgeOffloading = fullyOffloading * (1 - CE_[i:f, :])
            edgeOffloadings.append(edgeOffloading)

        for i in range(self.edge_no):
            f = i + 1

            # Pool cache indicator
            if self.edge_no > 1:
                poolCachingIndicator = edgeCachingIndicators.copy()
                poolCachingIndicator.pop(i)
                poolCache = poolCachingIndicator.pop(0)
                for Indicator in poolCachingIndicator:
                    poolCache = np.logical_or(poolCache, Indicator)
            else:
                poolCache = 0

            # Execution On the edge completely
            edge = cachedOnEdges[i]
            edge_exec_time += np.sum(edge * self.V2E_trans_time_[i*l:f*l, :])
            edge_exec_time += np.sum(edge * self.edge_exec_time)
            n_edge += np.sum(edge)

            # Service programs of some requests launched by moving vehicles
            # are not cached neither on the nearest edge nor on the vehicles
            pool = edgeOffloadings[i] * poolCache
            
            # for energy consumption
            E2ETransTime += np.sum(pool * self.E2E_trans_time)

            pool_exec_time += np.sum(pool * self.V2E_trans_time_[i*l:f*l, :])
            pool_exec_time += np.sum(pool * self.E2E_trans_time)
            pool_exec_time += np.sum(pool * self.edge_exec_time)
            n_pool += np.sum(pool)

            # Some requests launched by vehicles transmit to cloud
            # because corresponding service programs are not cached on edges
            cloud = edgeOffloadings[i] * (1 - poolCache)

            # for energy consumption
            E2CTransTime += np.sum(cloud * self.E2C_trans_time)

            # for delay
            cloud_exec_time += np.sum(cloud * self.V2E_trans_time_[i*l:f*l, :])
            cloud_exec_time += np.sum(cloud * self.E2C_trans_time)
            n_cloud += np.sum(cloud)

        nums = [n_local, n_edge, n_pool, n_cloud, task_no, n_v2e_partial, n_e2e_partial]
        
        sum_delay += (local_exec_time + edge_exec_time + pool_exec_time + cloud_exec_time)
                 
        execTimes = [local_exec_time, edge_exec_time, pool_exec_time, cloud_exec_time,
                     0, 0, sum_delay]
         
        transTimes = [E2ETransTime, E2CTransTime]
        
        deObs = [I, CV, B, X, CE_]

        return self.func(execTimes, nums, transTimes, deObs)
    
    def func(self, execTimes, nums, transTimes, deObs):
        
        sum_delay = execTimes[6]
        
        avg_delay = execTimes[6] / nums[4]
        
        # Energy
        total_energy =\
            self.enerty_cons_factor * execTimes[1] +\
            self.enerty_cons_factor * execTimes[2] +\
            self.edge_power * transTimes[0] +\
            self.cloud_power * transTimes[1]
              
        # Delay            
        for i in range(4):
            if nums[i] > 0:
                execTimes[i] /= nums[i]
                     
        total_delay = max(execTimes[0], execTimes[1], execTimes[2], execTimes[3], execTimes[4], execTimes[5])

        # Reward
        delay_reward  = -1 * (sum_delay - self.min_delay * nums[4]) / self.max_delay * nums[4]
        
        energy_reward = -1 * total_energy / self.max_energy_consu * nums[4]

        reward = self.epsilon * delay_reward + \
            (1 - self.epsilon) * energy_reward

        # done
        done = self.checkConvergency(total_delay)

        # info
        info = {}

        # Make New Observation
        self.observation_ = self.newObservation(CE_=deObs[4])
        
        observation_ = self.flattenObservation(self.observation_)
            
        observation = self.createNewFlattenObservation(deObs[0], deObs[1], deObs[2], deObs[3], deObs[4])       
        
        return  observation_, observation, avg_delay, total_delay, total_energy, reward, done, info, nums

    def checkConvergency(self, total_delay):
        done = False
        if total_delay <= self.convergence_delay:
            self.conv_cntr -= 1
        if self.conv_cntr <= 0:
            done = True
        return done

    def actionSampleDecode(self, action):
        # Action spase decode
        # The service caching indicator of edge node e
        # after completing all task offloading and calculation in time slot t.
        CE_ = []
        # The proportion of vehicle offloading tasks to edge nodes in time slot t.
        OV = []
        # The proportion of edge nodes offloading tasks to edge pool in time slot t.
        OE = []
        action_sample = [CE_, OV, OE]
        for i in range(len(action_sample)):
            for j in range(self.edge_no):
                action_sample[i].append(action[i][j])

        CE_ =\
            np.reshape(np.vstack(([[CE_.pop(0)] for _ in range(self.edge_no)])),
                       (self.edge_no, self.max_service_program))
        OV =\
            np.reshape(np.vstack(([[OV.pop(0)] for _ in range(self.edge_no)])),
                       (self.edge_no * self.max_vehicle_density, self.max_service_program))
        OE =\
            np.reshape(np.vstack(([[OE.pop(0)] for _ in range(self.edge_no)])),
                       (self.edge_no, self.max_service_program))

        return CE_, OV, OE

    def observationSampleDecode(self, observation):

        # Observation space decode
        # The request indicator for tasks by vehicles
        # within the coverage range of edge nodes at time slot t,
        I = []
        # The service caching indicator for vehicles
        # within the coverage range of edge nodes at time slot t
        CV = []
        # The bandwidth that edge nodes allocates to vehicles at time slot t
        B = []
        # The received SINR of edge nodes at time slot t
        X = []
        # The service caching indicator of edge nodes at time slot t
        CE = []
        observation_sample = [I, CV, B, X, CE]
        for i in range(len(observation_sample)):
            for j in range(self.edge_no):
                observation_sample[i].append(observation[i][j])

        # Frequency division multiplexing
        dens = 0
        mul = None
        for i in range(self.edge_no):
            dens = np.count_nonzero(np.sum(I[i], axis=1))
            mul = np.reshape(np.sum(I[i], axis=1),
                             (self.max_vehicle_density, 1))
            mul[mul != 0] = 1
            if dens > 0:
                B[i] = self.channel_bandwidth * (1 / dens)
                B[i] *= mul
            else:
                B[i] = (mul * 0)

        I =\
            np.reshape(np.vstack(([[I.pop(0)] for _ in range(self.edge_no)])),
                       (self.edge_no * self.max_vehicle_density, self.max_service_program))
        CV =\
            np.reshape(np.vstack(([[CV.pop(0)] for _ in range(self.edge_no)])),
                       (self.edge_no * self.max_vehicle_density, self.max_service_program))
        B =\
            np.reshape(np.vstack(([[B[i]] for i in range(self.edge_no)])),
                       (self.edge_no * self.max_vehicle_density, 1))
        X =\
            np.reshape(np.vstack(([[X.pop(0)] for _ in range(self.edge_no)])),
                       (self.edge_no * self.max_vehicle_density, 1))
        CE =\
            np.reshape(np.vstack(([[CE.pop(0)] for _ in range(self.edge_no)])),
                        (self.edge_no , self.max_service_program))

        return I, CV, B, X, CE

    def newObservation(self, CE_=None):

        observation = self.observation_space.sample()

        I = []
        B = []
        for j in range(self.edge_no):
            I.append(observation[0][j])
            B.append(observation[2][j])

        # Frequency division multiplexing
        dens = 0
        mul = None
        for i in range(self.edge_no):
            dens = np.count_nonzero(np.sum(I[i], axis=1))
            mul = np.reshape(np.sum(I[i], axis=1),
                             (self.max_vehicle_density, 1))
            mul[mul != 0] = 1
            if dens > 0:
                B[i] = self.channel_bandwidth * (1 / dens)
                B[i] *= mul
            else:
                B[i] = (mul * 0)

        for j in range(self.max_vehicle_density):
            for i in range(self.edge_no):
                observation[2][i][j] = B[i][j]

        # Update
        if CE_ is not None:
            for i in range(self.edge_no):
                f = i + 1
                observation[4][i][0]= CE_[i:f, :]

        return observation

    def flattenAction(self, action):
        CE_, OV, OE = self.actionSampleDecode(action)
        flattenAct = self.createNewFlattenAction(CE_, OV, OE)
        return flattenAct

    def flattenObservation(self, observation):
        I, CV, B, X, CE = self.observationSampleDecode(observation)
        flattenObs = self.createNewFlattenObservation(I, CV, B, X, CE)
        return flattenObs

    def createNewFlattenObservation(self, I, CV, B, X, CE_):
        item_list = []
        observation_ = [I, CV, B, X, CE_]
        for item in observation_:
            item_list.append(np.reshape(item, -1))

        flatten_observation_ = item_list.pop(0)
        for item in item_list:
            flatten_observation_ = np.hstack((flatten_observation_, item))

        return flatten_observation_
    
    def createNewFlattenAction(self, CE_, OV, OE):
        item_list = []
        action = [CE_, OV, OE]
        for item in action:
            item_list.append(np.reshape(item, -1))

        flatten_action = item_list.pop(0)
        for item in item_list:
            flatten_action = np.hstack((flatten_action, item))

        return flatten_action

    def extractAction(self, action):
        nk = self.max_service_program
        nv = self.max_vehicle_density
        ne = self.edge_no
        f = 0
        l = ne*nk
        CE_ = np.reshape(action[f:l], (ne, nk)).copy()
        
        # Constraint on storage capacity of edge caching
        for i in range(self.edge_no):
            ones = CE_[i][CE_[i] >= 0.5]
            while True:
                ones = ones[ones >= 0.5]
                if ones.size > self.edge_max_caching:
                    ones[np.argmin(ones)] = -1
                    CE_[i][CE_[i] >= 0.5] = ones
                else:
                    break
                
        CE_[CE_ < 0.5] = 0
        CE_[CE_ >= 0.5] = 1
        
        f = l
        l = l + nv*nk*ne

        OV = np.reshape(action[f:l], (ne*nv, nk)).copy()
        OV = clip_by_value(OV, self.min_offloading, self.max_offloading)
        
        f = l

        OE = np.reshape(action[l:], (ne, nk)).copy()
        OE = clip_by_value(OE, self.min_offloading, self.max_offloading)

        return CE_, OV, OE

    def render(self, render_mode="human"):
        "Nothing to coding"
        pass

    def close(self):
        self.conv_cntr = 10 * setup.N

    def seed(self):
        "Nothing to coding"
        pass
