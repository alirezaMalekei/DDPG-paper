import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from keras.initializers import HeNormal, GlorotNormal
from setup import LOAD_PATH

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=1024, fc2_dims=512, fc3_dims=256, fc4_dims=128, fc5_dims=128, 
                 fc6_dims=128, fc7_dims=128, n_actions=2, name='actor', chkpt_dir=LOAD_PATH):
        
        super(ActorNetwork, self).__init__()
        self.fc1_dims  = fc1_dims
        self.fc2_dims  = fc2_dims
        self.fc3_dims  = fc3_dims
        self.fc4_dims  = fc4_dims
        self.fc5_dims  = fc5_dims
        self.fc6_dims  = fc6_dims
        self.fc7_dims  = fc7_dims
        
        self.n_actions = n_actions
        
        self.checkpoint_dir  = chkpt_dir
        self.model_name      = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.keras')
        
        # I use initializer, why??? I’ve encountered some issues, not converging or converging slowly!!
        # Initializers in a Dense layer are crucial for setting the initial random weights of the layer. 
        # Proper initialization can significantly impact the training process and the performance of the neural network. 
        # Here are some key reasons why we use initializers:
        #     1. Avoiding Symmetry: If all weights are initialized to the same value (e.g., zero), the neurons in each layer will learn the same features during training. 
        #         This symmetry prevents the network from learning effectively. Initializers help break this symmetry by assigning different initial values to the weights.
        #     2. Preventing Vanishing/Exploding Gradients: Poor initialization can lead to vanishing or exploding gradients, where the gradients become too small or too large during backpropagation. 
        #         This can slow down or completely halt the training process. Proper initialization helps maintain a stable gradient flow.
        #     3. Faster Convergence: Properly initialized weights can help the model converge faster during training. 
        #         This means the model can reach an optimal solution more quickly, reducing the training time.
        #     4. Improved Performance: Initializers can improve the overall performance of the model by ensuring that the weights start with values that are conducive to learning. 
        #         This can lead to better accuracy and generalization.
        #     Different initializers are suited for different types of activation functions and network architectures. For example:
        #         Xavier (Glorot) Initialization: Works well with sigmoid, linear, and tanh activation functions.
        #         He Initialization: Designed for ReLU activation functions.
        #         LeCun Initialization: Suitable for activation functions like SELU.
        # and finally, Initializing the bias to zero is a common practice in neural networks for several reasons:
        #     1. Symmetry Breaking: While initializing weights with small random values helps break symmetry, biases can be safely initialized to zero without causing symmetry issues.
        #         This is because the small random weights already ensure that neurons learn different features.
        #     2. Simplification: Initializing biases to zero simplifies the initialization process. 
        #         Since biases are added to the weighted sum of inputs, starting them at zero allows the network to learn the necessary bias values during training.
        #     3. Stability: Zero initialization of biases can contribute to the stability of the training process. 
        #         It ensures that the initial output of neurons is not excessively large or small, which can help maintain a stable gradient flow.
        #     4. Common Practice: It is a widely accepted practice in the deep learning community 
        #         and is often used in conjunction with other weight initialization techniques like Xavier or He initialization.
        # By choosing the appropriate initializer, we can enhance the training process and achieve better results with your neural network.

        self.fc1 = Dense(self.fc1_dims, activation='relu', 
                          kernel_initializer=HeNormal(seed=42), bias_initializer='zeros'
                         )
        # self.dr1 = Dropout(0.25)
        
        self.fc2 = Dense(self.fc2_dims, activation='relu', 
                          kernel_initializer=HeNormal(seed=42), bias_initializer='zeros'
                         )
        # self.dr2 = Dropout(0.25)
        
        self.fc3 = Dense(self.fc3_dims, activation='relu', 
                          kernel_initializer=HeNormal(seed=42), bias_initializer='zeros'
                         )
        # self.dr3 = Dropout(0.25)
        
        self.fc4 = Dense(self.fc4_dims, activation='relu',
                          kernel_initializer=HeNormal(seed=42), bias_initializer='zeros'
                         )
        # self.dr4 = Dropout(0.25)

        # self.fc5 = Dense(self.fc5_dims, activation='relu', 
        #                   kernel_initializer=HeNormal(seed=42), bias_initializer='zeros')
        # self.dr5 = Dropout(0.25)
                
        # self.fc6 = Dense(self.fc6_dims, activation='relu', 
        #                  kernel_initializer=HeNormal(seed=42), bias_initializer='zeros')
        # self.dr6 = Dropout(0.25)
                
        # self.fc7 = Dense(self.fc7_dims, activation='relu', 
        #                  kernel_initializer=HeNormal(seed=42), bias_initializer='zeros')
        # self.dr7 = Dropout(0.25)
        
        self.mu  = Dense(self.n_actions, activation='sigmoid', 
                         kernel_initializer=GlorotNormal())
        
    def call(self, state):
        prob = self.fc1(state)
        # prob = self.dr1(prob)
        
        prob = self.fc2(prob)
        # prob = self.dr2(prob)
        
        prob = self.fc3(prob)
        # prob = self.dr3(prob)
        
        prob = self.fc4(prob)
        # prob = self.dr4(prob)
        
        # prob = self.fc5(prob)
        # prob = self.dr5(prob)
        
        # prob = self.fc6(prob)
        # prob = self.dr6(prob)
        
        # prob = self.fc7(prob)  
        # prob = self.dr7(prob)

        mu   = self.mu(prob)

        return mu