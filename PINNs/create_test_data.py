import numpy as np
import pandas as pd
from PINNs.Get_KKT_Lg import Get_KKT_Lg 

def create_test_data(simulation_parameters):
    n_bus = simulation_parameters['general']['n_bus']
    n_gbus = simulation_parameters['general']['n_gbus']
    n_line =simulation_parameters['general']['n_line']
    
    n_test_data_points = simulation_parameters['data_creation']['n_test_data_points']
    n_collocation = simulation_parameters['data_creation']['n_collocation']
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    n_total = n_data_points + n_collocation
    
    Lg_Max=simulation_parameters['Lg_Max']
    
    L_Val=pd.read_csv('Data_File/'+str(n_bus)+'/NN_input.csv').to_numpy()[n_total:n_total+n_test_data_points][:]
    L_type_data = np.ones((n_test_data_points, 1))

    x_test = [np.concatenate([L_Val], axis=0),
                  np.concatenate([L_type_data], axis=0)]
    
    results_v = pd.read_csv('Data_File/'+str(n_bus)+'/NN_output_V.csv').to_numpy()[n_total:n_total+n_test_data_points][:]
    y_volt_data = np.array(results_v) 

    results_pq = pd.read_csv('Data_File/'+str(n_bus)+'/NN_output_PQ.csv').to_numpy()[n_total:n_total+n_test_data_points][:]
    y_PQ_data = np.array(results_pq) 

    KKT_Lg=Get_KKT_Lg(n_bus,L_Val,results_pq,results_v)
    y_Lg = np.array(KKT_Lg)
    
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
    
    y_l_p_Lg  =y_Lg[:,l_p_s:l_p_e]/Lg_Max[0]
    y_mu_g_u_Lg=y_Lg[:,mu_g_u_s:mu_g_u_e]/Lg_Max[1]
    y_mu_g_d_Lg=y_Lg[:,mu_g_d_s:mu_g_d_e]/Lg_Max[2]
    y_mu_v_u_Lg=y_Lg[:,mu_v_u_s:mu_v_u_e]/Lg_Max[3]
    y_mu_v_d_Lg=y_Lg[:,mu_v_d_s:mu_v_d_e]/Lg_Max[4]
    y_mu_i_u_Lg=y_Lg[:,mu_i_u_s:mu_i_u_e]/Lg_Max[5]

    y_test= [y_volt_data, y_PQ_data, y_l_p_Lg,
                  y_mu_g_u_Lg, y_mu_g_d_Lg,
                  y_mu_v_u_Lg, y_mu_v_d_Lg, 
                  y_mu_i_u_Lg, np.zeros((n_test_data_points , 1))]

    np.savetxt('Test_output/'+str(n_bus)+'/features_test.csv',L_Val, fmt='%s', delimiter=',')
    np.savetxt('Test_output/'+str(n_bus)+'/labels_test.csv',y_PQ_data, fmt='%s', delimiter=',')
    return x_test, y_test