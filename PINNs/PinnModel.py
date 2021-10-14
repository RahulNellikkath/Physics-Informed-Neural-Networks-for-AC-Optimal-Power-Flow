from PINNs.PinnLayer import PinnLayer

import tensorflow as tf


class PinnModel(tf.keras.models.Model):

    def __init__(self,w1,w2,simulation_parameters):
        super(PinnModel, self).__init__()

        self.n_bus = simulation_parameters['general']['n_bus']
        self.PinnLayer = PinnLayer(simulation_parameters=simulation_parameters)
        n_d_p = simulation_parameters['data_creation']['n_data_points']
        n_c = simulation_parameters['data_creation']['n_collocation']
        n_lbus = simulation_parameters['general']['n_lbus']
        n_t = n_d_p + n_c

        loss_weights = [n_t/n_d_p,n_t/n_d_p,w1*10**-6,w1*10**-6,w1*10**-6,w1*10**-6,w1*10**-6,w1*10**-6,w2*10**-5]

        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.mean_absolute_error,
                     loss_weights=loss_weights)

        self.build(input_shape=[(None, 2*n_lbus), (None, 1)])

    def call(self, inputs, training=None, mask=None):
        L_val, x_type = inputs
        # NN Outputs
        n_o_v, n_o_pq, n_o_l_p, n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u, n_o_PINN = self.PinnLayer(L_val)

        loss_network_output_v = tf.multiply(n_o_v, x_type)
        loss_network_output_pq = tf.multiply(n_o_pq, x_type)
        
        loss_network_output_l_p = tf.multiply(n_o_l_p, x_type)
        loss_network_output_mu_g_u = tf.multiply(n_o_mu_g_u, x_type)
        loss_network_output_mu_g_d = tf.multiply(n_o_mu_g_d, x_type)
        loss_network_output_mu_v_u = tf.multiply(n_o_mu_v_u, x_type)
        loss_network_output_mu_v_d = tf.multiply(n_o_mu_v_d, x_type)
        loss_network_output_mu_i_u = tf.multiply(n_o_mu_i_u, x_type)
        
        loss_network_output_physics = n_o_PINN

        loss_output = (loss_network_output_v,
                       loss_network_output_pq,
                       loss_network_output_l_p,
                       loss_network_output_mu_g_u,
                       loss_network_output_mu_g_d,
                       loss_network_output_mu_v_u,
                       loss_network_output_mu_v_d,
                       loss_network_output_mu_i_u,
                       loss_network_output_physics)

        return loss_output
