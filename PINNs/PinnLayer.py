import tensorflow as tf
import numpy as np
from PINNs.DenseCoreNetwork import DenseCoreNetwork

    
class PinnLayer(tf.keras.layers.Layer):
    """
    This layer includes the prediction
    """

    def __init__(self, simulation_parameters):
        super(PinnLayer, self).__init__()

        self.n_bus = simulation_parameters['general']['n_bus']
        self.n_gbus = simulation_parameters['general']['n_gbus']
        self.n_line =simulation_parameters['general']['n_line']
        self.g_bus = simulation_parameters['general']['g_bus']
        self.neurons_in_hidden_layers_V = simulation_parameters['training']['neurons_in_hidden_layers_V']
        self.neurons_in_hidden_layers_G = simulation_parameters['training']['neurons_in_hidden_layers_G']
        self.neurons_in_hidden_layers_Lg = simulation_parameters['training']['neurons_in_hidden_layers_Lg']
        self.DenseCoreNetwork = DenseCoreNetwork(n_bus =self.n_bus, n_gbus =self.n_gbus, n_line=self.n_line,
                                                 neurons_in_hidden_layers_V=self.neurons_in_hidden_layers_V,
                                                 neurons_in_hidden_layers_G=self.neurons_in_hidden_layers_G,
                                                 neurons_in_hidden_layers_Lg=self.neurons_in_hidden_layers_Lg)

        self.C_Pg = simulation_parameters['true_system']['C_Pg']
        self.C_Qg = np.zeros((1, self.n_gbus))
        self.Gen_max = simulation_parameters['true_system']['Gen_max']
        self.Gen_min = simulation_parameters['true_system']['Gen_min']
        self.Map_g = simulation_parameters['true_system']['Map_g']
        self.Map_L = simulation_parameters['true_system']['Map_L']
        
        self.Vmax = simulation_parameters['true_system']['Vmax']
        self.Vmin = simulation_parameters['true_system']['Vmin']
        self.Lg_Max = simulation_parameters['Lg_Max']
        
        self.L_limit = simulation_parameters['true_system']['L_limit']
        self.Y = simulation_parameters['true_system']['Y']
        self.Yconj = simulation_parameters['true_system']['Yconj']
        self.Ybr = simulation_parameters['true_system']['Ybr']
        self.IM = simulation_parameters['true_system']['IM']

        
    def Get_KKT_error(self,Volt, P_Gens, P_Loads, n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u):       
        
        # Reference bus 
        KKT_error = tf.reduce_sum(tf.abs(Volt[:,self.n_bus + self.g_bus[0]-1]))

        # PowerFlow Equation
        for i in range(1,self.n_bus):
            M=np.zeros((2*self.n_bus,2*self.n_bus))
            M[i-1,i-1]=1
            M[self.n_bus+i-1,self.n_bus+i-1]=1
            H_p =M@self.Y 
            H_q =M@self.Yconj
            
            e_P = np.zeros((2*self.n_bus,1))
            e_P[i-1,0] = 1
            e_Q = np.zeros((2*self.n_bus,1))
            e_Q[self.n_bus+i-1,0] = 1      
            
            KKT_error = KKT_error + tf.reduce_sum(tf.transpose(tf.linalg.diag_part((Volt@H_p)@tf.transpose(Volt))) + ((P_Loads@self.Map_L)-(P_Gens@self.Map_g))@e_P,1)
            KKT_error = KKT_error + tf.reduce_sum(tf.transpose(tf.linalg.diag_part((Volt@H_q)@tf.transpose(Volt))) + ((P_Loads@self.Map_L) -(P_Gens@self.Map_g))@e_Q,1)
            
        # Generation Violation
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(P_Gens - self.Gen_max), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(self.Gen_min - P_Gens), axis=1)
        
        # Voltage violation
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.square(Volt[:,0:self.n_bus])+tf.square(Volt[:,self.n_bus:2*self.n_bus])-self.Vmax**2), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(self.Vmin**2-tf.square(Volt[:,0:self.n_bus])+tf.square(Volt[:,self.n_bus:2*self.n_bus])), axis=1)
                             
        # Line Flow Violation
        Ibr = tf.transpose((self.Ybr@(self.IM@tf.transpose(Volt))))
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.square(Ibr[:,0:self.n_line])+tf.square(Ibr[:,self.n_line:2*self.n_line]) - self.L_limit**2), axis=1)
        
        
        #KKT Conditions
        
        # Generation Violation
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(tf.multiply(n_o_mu_g_u,P_Gens - self.Gen_max)), axis=1)/self.n_gbus
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(tf.multiply(n_o_mu_g_d,self.Gen_min - P_Gens)), axis=1)/self.n_gbus
        
        # Voltage violation
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(tf.multiply(n_o_mu_v_u,tf.square(Volt[:,0:self.n_bus])+tf.square(Volt[:,self.n_bus:2*self.n_bus])-self.Vmax**2)), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(tf.multiply(n_o_mu_v_d,self.Vmin**2 - tf.square(Volt[:,0:self.n_bus])+tf.square(Volt[:,self.n_bus:2*self.n_bus]))), axis=1)
        
        # Line Flow Violation
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(tf.multiply(n_o_mu_i_u,tf.square(Ibr[:,0:self.n_line])+tf.square(Ibr[:,self.n_line:2*self.n_line]) - self.L_limit**2)), axis=1)
        
        # dL/ dP_Gen        
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(n_o_mu_g_u*self.Lg_Max[1]-n_o_mu_g_d*self.Lg_Max[2]+(n_o_l_p*self.Lg_Max[0])@np.transpose(self.Map_g) - np.concatenate((self.C_Pg,self.C_Qg), axis=1)), axis=1)
        
        #KKT dual variables
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.math.negative(n_o_mu_g_u)), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.math.negative(n_o_mu_g_d)), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.math.negative(n_o_mu_v_u)), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.math.negative(n_o_mu_v_d)), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.math.negative(n_o_mu_i_u)), axis=1)
        
        return KKT_error
        
    def call(self, inputs, **kwargs):
        
        L_Val = tf.convert_to_tensor(inputs)
        
        # Get NN Outputs
        n_o_v, n_o_pq, n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u = self.DenseCoreNetwork.call_inference(L_Val, **kwargs)
        
        # Get KKT Error
        KKT_error = self.Get_KKT_error(n_o_v, n_o_pq, L_Val, n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u)
        
        return n_o_v, n_o_pq, n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u, KKT_error

