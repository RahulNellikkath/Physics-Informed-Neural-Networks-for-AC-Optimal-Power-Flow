0# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:58:24 2021

@author: rnelli
"""

import numpy as np
from PINNs.create_example_parameters import create_example_parameters

def Get_KKT_Lg(n_bus,P_Loads,P_Gens,Volts):
    
    simulation_parameters = create_example_parameters(n_bus)
    
    n_line =simulation_parameters['general']['n_line']
    n_gbus=simulation_parameters['general']['n_gbus']    
    n_lbus=simulation_parameters['general']['n_lbus']

    g_bus = simulation_parameters['general']['g_bus']
    Y=simulation_parameters['true_system']['Y']
    Yconj=simulation_parameters['true_system']['Yconj']
    Ybr = simulation_parameters['true_system']['Ybr']
    IM=simulation_parameters['true_system']['IM']
    L_limit=np.transpose(simulation_parameters['true_system']['L_limit'])
    C_Pg=simulation_parameters['true_system']['C_Pg']
    
    Gen_max=simulation_parameters['true_system']['Gen_max']
    Gen_min=simulation_parameters['true_system']['Gen_min'] 
    Vmin=simulation_parameters['true_system']['Vmin'] 
    Vmax=simulation_parameters['true_system']['Vmax']

    Lg_val=[]
    
    for d in range(np.size(P_Loads,0)):
        P_Gen=np.zeros((1,2*n_gbus))
        P_Gen=P_Gens[d].reshape(1,2*n_gbus)
        
        V = np.zeros((1,2*n_bus))
        V = Volts[d]

        # X= [lambda_p(2*n_bus), mu_g_up(2*n_gbus), mu_g_dn(2*n_gbus), 
        #       mu_v_up(n_bus), mu_v_dn(n_bus), mu_i_up(n_line)]'

        lambda_p_srt= 0
        
        mu_g_up_srt= 2*n_bus
        mu_g_dn_srt= 2*n_bus+2*n_gbus

        mu_v_up_srt= 2*n_bus+4*n_gbus
        mu_v_dn_srt= 3*n_bus+4*n_gbus
        
        mu_i_up_srt= 4*n_bus+4*n_gbus
        mu_i_up_end= 4*n_bus+4*n_gbus+n_line

        
        # Total Num of variables = 4*n_bus+4*n_gbus+n_line 
        # Maximum possible no of equations = 4*n_bus+6*n_gbus+n_line
        
        A=np.zeros([4*n_bus+6*n_gbus+n_line,4*n_bus+4*n_gbus+n_line])
        B=np.zeros([4*n_bus+6*n_gbus+n_line,1]) 
        
        # derivative of stationarity conditions w.r.t V Part 1 (Load balance)
        for b in range(n_bus):
            M=np.zeros((2*n_bus,2*n_bus))
            M[b,b]=1
            M[n_bus+b-1,n_bus+b-1]=1
            H_p =(M@Y) #+ Yconj@M)/2
            H_q =(M@Yconj) #- Yconj@M)/2
            # Real power 
            A[0:2*n_bus,lambda_p_srt+b:lambda_p_srt+b+1] = (V@H_p).reshape(2*n_bus,1) 
            # Reactive power
            A[0:2*n_bus,lambda_p_srt+n_bus+b:lambda_p_srt+n_bus+b+1] = (V@H_q).reshape(2*n_bus,1)            
       
        # derivative of stationarity conditions w.r.t V Part 2 (Voltage Limits)
        for b in range(n_bus):
            M=np.zeros((2*n_bus,2*n_bus))
            M[b,b]=1
            M[n_bus+b,n_bus+b]=1
            A[0:2*n_bus,mu_v_up_srt+b:mu_v_up_srt+b+1] = (V@M).reshape(2*n_bus,1)
            A[0:2*n_bus,mu_v_dn_srt+b:mu_v_dn_srt+b+1] = (V@(-M)).reshape(2*n_bus,1)
        
        # derivative of stationarity conditions w.r.t V Part 3 (Line Flow Limits)
        Ib=np.zeros((2*n_line,2*n_bus))
        Ib=(Ybr@IM).reshape(2*n_line,2*n_bus)
        for l in range(n_line):
            M=np.zeros((2*n_line,2*n_line))
            M[l,l]=1
            M[n_line+l,n_line+l]=1          
            A[0:2*n_bus,mu_i_up_srt+l:mu_i_up_srt+l+1]=((np.transpose(Ib)@(M@Ib))@V).reshape(2*n_bus,1)
            
        row=2*n_bus
        
        # derivative of stationarity conditions w.r.t G 
        i=0
        for g in g_bus:
            # P
            A[row,lambda_p_srt+(g-1)] = 1 # From Load balance
            A[row,mu_g_up_srt+i] = 1 #Gen Limits
            A[row,mu_g_dn_srt+i] = -1 #Gen Limits
            B[row,0] = C_Pg[0][i] #Gen Cost active
            row+=1
            # Q
            A[row,lambda_p_srt+n_bus+(g-1)] = 1 # From Load balance
            A[row,mu_g_up_srt+n_gbus+i] = 1 #Gen Limits
            A[row,mu_g_dn_srt+n_gbus+i] = -1#Gen Limits
            B[row,0] = 0 #C_Pg[0][i] #Gen Cost reactive
            i+=1
            row+=1
            
        # Generation
        epsi=0.001 #(a 0.1 % of max value is considered to compensate the calculation errors)
        for g in range(2*n_gbus):
            if (1-epsi)*Gen_max[0][g] - P_Gen[0][g]  > 0 :
                A[row,mu_g_up_srt+g] = Gen_max[0][g] - P_Gen[0][g]
                row+=1
            if Gen_min[0][g] > 0:
                if P_Gen[0][g] - (1+epsi)*Gen_min[0][g] > 0 :
                    A[row,mu_g_dn_srt+g] = P_Gen[0][g] - Gen_min[0][g]
                    row+=1
            elif Gen_min[0][g] < 0:
                 if P_Gen[0][g] - (1-epsi)*Gen_min[0][g] > 0 :
                    A[row,mu_g_dn_srt+g] = P_Gen[0][g] - Gen_min[0][g]
                    row+=1
            else:
                if P_Gen[0][g] > epsi*Gen_max[0][g]:
                    A[row,mu_g_dn_srt+g] = P_Gen[0][g]
                    row+=1
        
        epsi=0.001 #(a 0.1 % of max value is considered to compensate the calculation errors)
        # Voltage
        Vsqr=V[0:n_bus]**2 + V[n_bus:2*n_bus]**2
        for b in range(n_bus):
            M=np.zeros((2*n_bus,2*n_bus))
            M[b,b]=1
            M[n_bus+b,n_bus+b]=1
            if ((1-epsi)*Vmax*Vmax - ((V@M)@np.transpose(V)) > 0):
                A[row,mu_v_up_srt+b] = (Vmax*Vmax - ((V@M)@np.transpose(V)))
                row+=1
            if (((V@M)@np.transpose(V)) - Vmin*Vmin*(1+epsi) > 0 ):
                A[row,mu_v_dn_srt+b] = ((V@M)@np.transpose(V)) - Vmin*Vmin
                row+=1   
        
        epsi=0.001 #(a 0.1 % of max value is considered to compensate the calculation errors)  
        Ibr = Ybr@(IM@V)
        for l in range(n_line):
            M=np.zeros((2*n_line,2*n_line))
            M[l,l]=1
            M[n_line+l,n_line+l]=1
            if ((L_limit[l]**2)*(1-epsi) - ((Ibr@M)@np.transpose(Ibr)) > 0):
                A[row,mu_i_up_srt+l] = 1
                row+=1

        x=np.linalg.lstsq(A,B,rcond=None)
        Lg=x[0]
        Lg_val.append(Lg.reshape(1,4*n_bus+4*n_gbus+n_line)[0])
        # print('rank',np.linalg.matrix_rank(A))  
    return Lg_val
        