import numpy as np
import pandas as pd
from PINNs.Get_KKT_Lg import Get_KKT_Lg 

def create_data(simulation_parameters):
    
    n_bus = simulation_parameters['general']['n_bus']
    n_gbus = simulation_parameters['general']['n_gbus']
    n_line =simulation_parameters['general']['n_line']
    n_gbus=simulation_parameters['general']['n_gbus'] 
    
    n_collocation = simulation_parameters['data_creation']['n_collocation']
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    n_total = n_data_points + n_collocation

    L_Val=pd.read_csv('Data_File/'+str(n_bus)+'/NN_input.csv').to_numpy()[0:n_data_points+n_collocation][:] 
    L_type_collocation = np.zeros((n_collocation, 1))
    L_type_data = np.ones((n_data_points, 1))
    x_training = [L_Val,
                  np.concatenate([L_type_data, L_type_collocation], axis=0)]
    

    results_v = pd.read_csv('Data_File/'+str(n_bus)+'/NN_output_V.csv').to_numpy()[0:n_data_points][:]
    y_volt_data = results_v 
    y_volt_collocation = np.zeros((n_collocation, 2*n_bus))
    y_volt = np.concatenate([y_volt_data, y_volt_collocation], axis=0)
    
    results_pq = pd.read_csv('Data_File/'+str(n_bus)+'/NN_output_PQ.csv').to_numpy()[0:n_data_points][:]
    y_PQ_data = results_pq
    
    y_PQ_collocation = np.zeros((n_collocation, 2*n_gbus))
    y_PQ = np.concatenate([y_PQ_data, y_PQ_collocation], axis=0)
    
    KKT_Lg=Get_KKT_Lg(n_bus,L_Val[0:n_data_points][:],results_pq,results_v)
    y_Lg_data = KKT_Lg
    y_Lg_collocation = np.zeros((n_collocation, 4*n_bus+4*n_gbus+n_line))
    y_Lg = np.concatenate([y_Lg_data, y_Lg_collocation], axis=0)

   
    l_p_s= 0
    l_p_e= 2*n_bus
    
    mu_g_u_s= 2*n_bus
    mu_g_u_e= 2*n_bus+2*n_gbus
    mu_g_d_s= 2*n_bus+2*n_gbus
    mu_g_d_e= 2*n_bus+4*n_gbus

    mu_v_u_s= 2*n_bus+4*n_gbus
    mu_v_u_e= 3*n_bus+4*n_gbus
    mu_v_d_s= 3*n_bus+4*n_gbus
    mu_v_d_e= 4*n_bus+4*n_gbus
    
    mu_i_u_s= 4*n_bus+4*n_gbus
    mu_i_u_e= 4*n_bus+4*n_gbus+n_line

    
    
    Lg_Max=[]
    
    lg_max=np.max(np.abs(y_Lg[:,l_p_s:l_p_e]),axis=0).reshape((1, l_p_s-l_p_e))+1e-5 #adding one to avoid division by zero when max of Lg is zero
    y_l_p_Lg  =y_Lg[:,l_p_s:l_p_e]/lg_max
    Lg_Max.append(lg_max)
    
    lg_max=np.max(np.abs(y_Lg[:,mu_g_u_s:mu_g_u_e]),axis=0).reshape((1, mu_g_u_e-mu_g_u_s))+1e-5
    y_mu_g_u_Lg=y_Lg[:,mu_g_u_s:mu_g_u_e]/lg_max
    Lg_Max.append(lg_max)
    
    lg_max=np.max(np.abs(y_Lg[:,mu_g_d_s:mu_g_d_e]),axis=0).reshape((1, mu_g_d_s-mu_g_d_e))+1e-5
    y_mu_g_d_Lg=y_Lg[:,mu_g_d_s:mu_g_d_e]/lg_max
    Lg_Max.append(lg_max)
    
    lg_max=np.max(np.abs(y_Lg[:,mu_v_u_s:mu_v_u_e]),axis=0).reshape((1, mu_v_u_s-mu_v_u_e))+1e-5
    y_mu_v_u_Lg=y_Lg[:,mu_v_u_s:mu_v_u_e]/lg_max
    Lg_Max.append(lg_max)
    
    lg_max=np.max(np.abs(y_Lg[:,mu_v_d_s:mu_v_d_e]),axis=0).reshape((1, mu_v_d_s-mu_v_d_e))+1e-5
    y_mu_v_d_Lg=y_Lg[:,mu_v_d_s:mu_v_d_e]/lg_max
    Lg_Max.append(lg_max)

    lg_max=np.max(np.abs(y_Lg[:,mu_i_u_s:mu_i_u_e]),axis=0).reshape((1, mu_i_u_s-mu_i_u_e))+1e-5
    y_mu_i_u_Lg=y_Lg[:,mu_i_u_s:mu_i_u_e]/lg_max
    Lg_Max.append(lg_max)
    
    y_training = [y_volt, y_PQ, 
                  y_l_p_Lg,
                  y_mu_g_u_Lg, y_mu_g_d_Lg,
                  y_mu_v_u_Lg, y_mu_v_d_Lg, 
                  y_mu_i_u_Lg, np.zeros((n_total, 1))]

    np.savetxt('Test_output/'+str(n_bus)+'/features_train.csv',L_Val[0:n_data_points][:], fmt='%s', delimiter=',')
    np.savetxt('Test_output/'+str(n_bus)+'/labels_train.csv',np.array(results_pq), fmt='%s', delimiter=',')
    return x_training, y_training, Lg_Max
