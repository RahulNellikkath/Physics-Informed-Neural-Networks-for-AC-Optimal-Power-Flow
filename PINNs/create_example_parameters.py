import numpy as np
from numpy import genfromtxt
import pandas as pd

def create_example_parameters(n_bus: int):
 
    if n_bus == 39:
        Gen = pd.read_csv('Data_File/39/Gen_39.csv',index_col=0)
        g_bus=Gen.index[Gen['Pg_max']!=0].to_numpy()
        n_gbus=len(g_bus)
        Pg_max=Gen['Pg_max'].to_numpy().reshape((1, n_gbus))
        Pg_min=Gen['Pg_min'].to_numpy().reshape((1, n_gbus))
        Qg_max=Gen['Qg_max'].to_numpy().reshape((1, n_gbus))
        Qg_min=Gen['Qg_min'].to_numpy().reshape((1, n_gbus))
        Gen_max=np.concatenate((Pg_max,Qg_max),axis=1).reshape(1, 2*n_gbus)
        Gen_min=np.concatenate((Pg_min,Qg_min),axis=1).reshape(1, 2*n_gbus)
        C_Pg=Gen['C_Pg'].to_numpy().reshape((1, n_gbus))
        # Map generator to bus 
        Map_g = np.zeros((2*n_gbus,2*n_bus))
        gen_no=0
        for g in g_bus:
            Map_g[gen_no][g-1]=1
            Map_g[n_gbus + gen_no][n_bus + g-1]=1
            gen_no+=1
        In_Map_g=np.linalg.pinv(Map_g)  
        Bus = pd.read_csv('Data_File/39/Bus_39.csv')
        l_bus=Bus['Node'].to_numpy()
        n_lbus=len(l_bus)
        Vmin=0.94
        Vmax=1.06
        # Map Loads to bus 
        Map_L = np.zeros((2*n_lbus,2*n_bus))
        l_no=0
        for l in l_bus:
            Map_L[l_no][l-1]=1
            Map_L[n_lbus+l_no][n_bus+l-1]=1
            l_no+=1
        Y= pd.read_csv('Data_File/39/Y.csv', header=None).to_numpy()
        Yconj= pd.read_csv('Data_File/39/Yconj.csv', header=None).to_numpy()
        Ybr= pd.read_csv('Data_File/39/Ybr.csv', header=None).to_numpy()
        IM= pd.read_csv('Data_File/39/IM.csv', header=None).to_numpy()
        n_line=int(np.size(IM,0)/2)
        L_limit= pd.read_csv('Data_File/39/L_limit.csv', header=None).to_numpy().reshape(1, n_line)

    if n_bus == 118:
        Gen = pd.read_csv('Data_File/118/Gen_118.csv',index_col=0)
        g_bus=Gen.index[Gen['Pg_max']!=0].to_numpy()
        n_gbus=len(g_bus)
        Pg_max=Gen['Pg_max'].to_numpy().reshape((1, n_gbus))
        Pg_min=Gen['Pg_min'].to_numpy().reshape((1, n_gbus))
        Qg_max=Gen['Qg_max'].to_numpy().reshape((1, n_gbus))
        Qg_min=Gen['Qg_min'].to_numpy().reshape((1, n_gbus))
        Gen_max=np.concatenate((Pg_max,Qg_max),axis=1).reshape(1, 2*n_gbus)
        Gen_min=np.concatenate((Pg_min,Qg_min),axis=1).reshape(1, 2*n_gbus)
        C_Pg=Gen['C_Pg'].to_numpy().reshape((1, n_gbus))
        
        Map_g = np.zeros((2*n_gbus,2*n_bus))
        gen_no=0
        for g in g_bus:
            Map_g[gen_no][g-1]=1
            Map_g[n_gbus + gen_no][n_bus + g-1]=1
            gen_no+=1
        In_Map_g=np.linalg.pinv(Map_g)  
        Bus = pd.read_csv('Data_File/118/Bus_118.csv')
        l_bus=Bus['Node'].to_numpy()
        n_lbus=len(l_bus)
        Vmin=0.94
        Vmax=1.06
        Map_L = np.zeros((2*n_lbus,2*n_bus))
        l_no=0
        for l in l_bus:
            Map_L[l_no][l-1]=1
            Map_L[n_lbus+l_no][n_bus+l-1]=1
            l_no+=1
        Y= pd.read_csv('Data_File/118/Y.csv', header=None).to_numpy()
        Yconj= pd.read_csv('Data_File/118/Yconj.csv', header=None).to_numpy()
        Ybr= pd.read_csv('Data_File/118/Ybr.csv', header=None).to_numpy()
        IM= pd.read_csv('Data_File/118/IM.csv', header=None).to_numpy()
        n_line=int(np.size(IM,0)/2)
        L_limit= pd.read_csv('Data_File/118/L_limit.csv', header=None).to_numpy().reshape(1, n_line)

    if n_bus == 162:
        Gen = pd.read_csv('Data_File/162/Gen_162.csv',index_col=0)
        g_bus=Gen.index[Gen['Pg_max']!=0].to_numpy()
        n_gbus=len(g_bus)
        Pg_max=Gen['Pg_max'].to_numpy().reshape((1, n_gbus))
        Pg_min=Gen['Pg_min'].to_numpy().reshape((1, n_gbus))
        Qg_max=Gen['Qg_max'].to_numpy().reshape((1, n_gbus))
        Qg_min=Gen['Qg_min'].to_numpy().reshape((1, n_gbus))
        Gen_max=np.concatenate((Pg_max,Qg_max),axis=1).reshape(1, 2*n_gbus)
        Gen_min=np.concatenate((Pg_min,Qg_min),axis=1).reshape(1, 2*n_gbus)
        C_Pg=Gen['C_Pg'].to_numpy().reshape((1, n_gbus))
        
        Map_g = np.zeros((2*n_gbus,2*n_bus))
        gen_no=0
        for g in g_bus:
            Map_g[gen_no][g-1]=1
            Map_g[n_gbus + gen_no][n_bus + g-1]=1
            gen_no+=1
        In_Map_g=np.linalg.pinv(Map_g)  
        Bus = pd.read_csv('Data_File/162/Bus_162.csv')
        l_bus=Bus['Node'].to_numpy()
        n_lbus=len(l_bus)
        Vmin=0.94
        Vmax=1.06
        Map_L = np.zeros((2*n_lbus,2*n_bus))
        l_no=0
        for l in l_bus:
            Map_L[l_no][l-1]=1
            Map_L[n_lbus+l_no][n_bus+l-1]=1
            l_no+=1
        Y= pd.read_csv('Data_File/162/Y.csv', header=None).to_numpy()
        Yconj= pd.read_csv('Data_File/162/Yconj.csv', header=None).to_numpy()
        Ybr= pd.read_csv('Data_File/162/Ybr.csv', header=None).to_numpy()
        IM= pd.read_csv('Data_File/162/IM.csv', header=None).to_numpy()
        n_line=int(np.size(IM,0)/2)
        L_limit= pd.read_csv('Data_File/162/L_limit.csv', header=None).to_numpy().reshape(1, n_line)

    # -----------------------------------------------------------------------------------------------
    # system parameters of the power system
    # Gen_max: Gen max cap in p.u.
    # Gen_min: Gen min cap in p.u.
    # Vmin: Voltage min in p.u.
    # Vmax: Voltage max in p.u.
    # L_limit: Line flow limit in p.u.
    # Y: Nodal admittance matrix
    # Yconj: Conjugate Nodal admittance matrix
    # Ybr: Branch admittance matrix
    # IM: Incidence Matrix
    # C_Pg: Cost of generation
    # Map_g: Map generator to bus
    # Map_L: Map Load to bus
    # g_bus: integer Bus IDs of Gen in the system
    # -----------------------------------------------------------------------------------------------
        
    true_system_parameters = {'Gen_max': Gen_max,
                              'Gen_min': Gen_min,
                              'Vmin':Vmin,
                              'Vmax':Vmax,
                              'L_limit':L_limit,
                              'Y':Y,
                              'Yconj':Yconj,
                              'Ybr':Ybr,
                              'IM':IM,
                              'C_Pg':C_Pg,
                              'Map_g':Map_g,
                              'Map_L':Map_L}

    # -----------------------------------------------------------------------------------------------
    # general parameters of the power system that are assumed to be known in the identification process
    # n_bus: integer number of buses in the system
    # n_lbus: integer number of loads in the system
    # n_gbus: integer number of generators in the system
    # n_line: integer number of lines in the system
    # l_bus: integer Bus IDs of loads in the system
    # g_bus: integer Bus IDs of Gen in the system
    # -----------------------------------------------------------------------------------------------

    
    general_parameters = {'n_bus': n_bus,
                          'n_lbus': n_lbus,
                          'n_gbus':n_gbus,
                          'n_line':n_line,
                          'l_bus': l_bus,
                          'g_bus': g_bus
                          }

    # -----------------------------------------------------------------------------------------------
    # parameters for the training data creation 
    # n_data_points: number of data points where measurements are present
    # n_collocation_points: number of points where the physics are evaluated at (additional to the data points)
    # -----------------------------------------------------------------------------------------------
    n_data_points = 2000
    n_test_data_points=2999
    n_collocation_points = 5000

    data_creation_parameters = {'n_data_points': n_data_points,
                                'n_collocation': n_collocation_points,
                                'n_test_data_points': n_test_data_points}

    # -----------------------------------------------------------------------------------------------
    # parameters for the scheduled training process and the network architecture
    # epoch_schedule: number of epochs per batch size
    # batching_schedule: batch size
    # neurons_in_hidden_layers: number of neurons for each hidden layer
    # -----------------------------------------------------------------------------------------------
    n_total = n_data_points + n_collocation_points


    epoch_schedule = [1000]

    batching_schedule = [int(np.ceil(n_total / 200))]
                         
    training_parameters = {'epoch_schedule': epoch_schedule,
                           'batching_schedule': batching_schedule,
                           'neurons_in_hidden_layers_V': [40,40,40],
                           'neurons_in_hidden_layers_G': [20,20,20],
                           'neurons_in_hidden_layers_Lg': [50,50,50]}

    # -----------------------------------------------------------------------------------------------
    # combining all parameters in a single dictionary
    # -----------------------------------------------------------------------------------------------
    simulation_parameters = {'true_system': true_system_parameters,
                             'general': general_parameters,
                             'data_creation': data_creation_parameters,
                             'training': training_parameters}

    return simulation_parameters
