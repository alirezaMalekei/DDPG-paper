import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import multiprocessing

from simu1 import simulation1
from simu2 import simulation2
from simu3 import simulation3

if __name__ == '__main__':
    
    """
    'D' delay minimization
    'O' no partial offloading
    'E' energy minimization
    'LRU' last recently used
    'R' random edge caching
    'F' no edge caching
    'H' executing all task in the cloud
    """
    try:
        # Delays Curve
        # DDPG-base edge caching aimed to delay minimization
        # DDPG-base edge caching with no partial offloading 
        # DDPG-base edge caching aimed to energy minimization
        # Cloud Computing approach aimed to computing all tasks by cloud
        # No Edge Caching approach
        processes = []
        process = multiprocessing.Process(target=simulation1())
        processes.append(process)
        process.start()
            
        # Delays per Density (clusterd column)
        process = multiprocessing.Process(target=simulation2())
        processes.append(process)
        process.start()
            
        # Delays for different Task Size
        process = multiprocessing.Process(target=simulation3())
        processes.append(process)
        process.start()
    
        for process in processes:
            process.join()
        
    except BaseException:
        print('...something goes wrong!')
    