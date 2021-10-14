# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:12:38 2021

@author: Rahul N
"""

import tensorflow as tf

class DenseCoreNetwork(tf.keras.models.Model):
    """
    This constitutes the core neural network with the PINN model. It outputs Voltage, GenValues and Dual Variables.
    """

    def __init__(self, n_bus, n_gbus, n_line, neurons_in_hidden_layers_V, neurons_in_hidden_layers_G, neurons_in_hidden_layers_Lg):

        super(DenseCoreNetwork, self).__init__()
        
# Hidden Layers V
        self.hidden_layer_V_0 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_V[0],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='first_layer')
        self.hidden_layer_V_1 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_V[1],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_V_1')
        self.hidden_layer_V_2 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_V[2],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_V_2')
                                                    
        self.dense_output_layer_V = tf.keras.layers.Dense(units=2*n_bus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_V')
        
# Hidden Layers G
        self.hidden_layer_G_0 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_G[0],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_G_0')
        self.hidden_layer_G_1 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_G[1],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_G_1')
        self.hidden_layer_G_2 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_G[2],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_G_2')
        self.dense_output_layer_PQ = tf.keras.layers.Dense(units=2*n_gbus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_l')

# Hidden Layers Lg(Dual Var)
        self.hidden_layer_Lg_0 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_Lg[0],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_Lg_0')
        self.hidden_layer_Lg_1 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_Lg[1],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_G_1')
        self.hidden_layer_Lg_2 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_Lg[2],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_G_2')
        self.dense_output_layer_l_p = tf.keras.layers.Dense(units=2*n_bus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_l')
        self.dense_output_layer_mu_g_u = tf.keras.layers.Dense(units=2*n_gbus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_l')
        self.dense_output_layer_mu_g_d = tf.keras.layers.Dense(units=2*n_gbus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_l')
        self.dense_output_layer_mu_v_u = tf.keras.layers.Dense(units=n_bus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_l')
        self.dense_output_layer_mu_v_d = tf.keras.layers.Dense(units=n_bus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_l')
        self.dense_output_layer_mu_i_u = tf.keras.layers.Dense(units=n_line,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_l')


    def call(self, inputs, training=None, mask=None):
        x_power = inputs
        # Hidden Layers V
        hidden_layer_V_0_output = self.hidden_layer_V_0(x_power)
        hidden_layer_V_1_output = self.hidden_layer_V_1(hidden_layer_V_0_output)
        hidden_layer_V_2_output = self.hidden_layer_V_2(hidden_layer_V_1_output)
        # NN Outputs Voltage
        n_o_v = self.dense_output_layer_V(hidden_layer_V_2_output)
        
        # Hidden Layers G
        hidden_layer_G_0_output = self.hidden_layer_G_0(x_power)
        hidden_layer_G_1_output = self.hidden_layer_G_1(hidden_layer_G_0_output)
        hidden_layer_G_2_output = self.hidden_layer_G_2(hidden_layer_G_1_output)
        # NN Outputs Gen
        n_o_pq = self.dense_output_layer_PQ(hidden_layer_G_2_output)
        
        # Hidden Layers Lg
        hidden_layer_Lg_0_output = self.hidden_layer_Lg_0(x_power)
        hidden_layer_Lg_1_output = self.hidden_layer_Lg_1(hidden_layer_Lg_0_output)
        hidden_layer_Lg_2_output = self.hidden_layer_Lg_2(hidden_layer_Lg_1_output)
        # NN Outputs Dual Var
        n_o_l_p = self.dense_output_layer_l_p(hidden_layer_Lg_2_output)
        n_o_mu_g_u = self.dense_output_layer_mu_g_u(hidden_layer_Lg_2_output)
        n_o_mu_g_d = self.dense_output_layer_mu_g_d(hidden_layer_Lg_2_output)
        n_o_mu_v_u = self.dense_output_layer_mu_v_u(hidden_layer_Lg_2_output)
        n_o_mu_v_d = self.dense_output_layer_mu_v_d(hidden_layer_Lg_2_output)
        n_o_mu_i_u = self.dense_output_layer_mu_i_u(hidden_layer_Lg_2_output)
        
        return n_o_v, n_o_pq, n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u
    
    def call_inference(self, inputs, training=None, mask=None):
        x_power = inputs
        # Hidden Layers V
        hidden_layer_V_0_output = self.hidden_layer_V_0(x_power)
        hidden_layer_V_1_output = self.hidden_layer_V_1(hidden_layer_V_0_output)
        hidden_layer_V_2_output = self.hidden_layer_V_2(hidden_layer_V_1_output)
        # NN Outputs Voltage
        n_o_v = self.dense_output_layer_V(hidden_layer_V_2_output)
        
        # Hidden Layers G
        hidden_layer_G_0_output = self.hidden_layer_G_0(x_power)
        hidden_layer_G_1_output = self.hidden_layer_G_1(hidden_layer_G_0_output)
        hidden_layer_G_2_output = self.hidden_layer_G_2(hidden_layer_G_1_output)
        # NN Outputs Gen
        n_o_pq = self.dense_output_layer_PQ(hidden_layer_G_2_output)
        
        # Hidden Layers Lg
        hidden_layer_Lg_0_output = self.hidden_layer_Lg_0(x_power)
        hidden_layer_Lg_1_output = self.hidden_layer_Lg_1(hidden_layer_Lg_0_output)
        hidden_layer_Lg_2_output = self.hidden_layer_Lg_2(hidden_layer_Lg_1_output)
        # NN Outputs Dual Var
        n_o_l_p = self.dense_output_layer_l_p(hidden_layer_Lg_2_output)
        n_o_mu_g_u = self.dense_output_layer_mu_g_u(hidden_layer_Lg_2_output)
        n_o_mu_g_d = self.dense_output_layer_mu_g_d(hidden_layer_Lg_2_output)
        n_o_mu_v_u = self.dense_output_layer_mu_v_u(hidden_layer_Lg_2_output)
        n_o_mu_v_d = self.dense_output_layer_mu_v_d(hidden_layer_Lg_2_output)
        n_o_mu_i_u = self.dense_output_layer_mu_i_u(hidden_layer_Lg_2_output)
         
        return n_o_v, n_o_pq, n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u
