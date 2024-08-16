import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from keras.initializers import HeNormal, GlorotNormal
from setup import LOAD_PATH

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=1024, fc2_dims=512, fc3_dims=256, fc4_dims=128, fc5_dims=128, 
                 fc6_dims=128, fc7_dims=64,name='critic', chkpt_dir=LOAD_PATH):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.fc6_dims = fc6_dims
        self.fc7_dims = fc7_dims
        
        self.checkpoint_dir  = chkpt_dir
        self.model_name      = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.keras')

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

        self.q   = Dense(1, activation='linear', 
                         kernel_initializer=GlorotNormal())

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        # action_value = self.dr1(action_value)
        
        action_value = self.fc2(action_value)
        # action_value = self.dr2(action_value)
        
        action_value = self.fc3(action_value)
        # action_value = self.dr3(action_value)
        
        action_value = self.fc4(action_value)
        # action_value = self.dr4(action_value)
        
        # action_value = self.fc5(action_value)
        # action_value = self.dr5(action_value)
        
        # action_value = self.fc6(action_value)
        # action_value = self.dr6(action_value)
        
        # action_value = self.fc7(action_value)
        # action_value = self.dr7(action_value)

        q            = self.q(action_value)

        return q
