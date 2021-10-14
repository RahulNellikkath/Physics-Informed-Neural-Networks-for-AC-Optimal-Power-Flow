import numpy as np
import time

from PINNs.create_example_parameters import create_example_parameters
from PINNs.create_data import create_data
from PINNs.PinnModel import PinnModel
from PINNs.create_test_data import create_test_data


def run_system_identification():
    n_bus=39
    simulation_parameters = create_example_parameters(n_bus)

    x_training, y_training, Lg_Max = create_data(simulation_parameters=simulation_parameters)

    simulation_parameters.update({'Lg_Max':Lg_Max})
    
    x_test, y_test = create_test_data(simulation_parameters=simulation_parameters)
    
    # Weightages of Lg and PINN
    lw1=0.05
    lw2=0.05
    model = PinnModel(lw1,lw2,simulation_parameters=simulation_parameters)
    np.set_printoptions(precision=3)
    print('Starting training')
    total_start_time = time.time()

    for n_epochs, batch_size in zip(simulation_parameters['training']['epoch_schedule'],
                                simulation_parameters['training']['batching_schedule']):

        epoch_start_time = time.time()
        model.fit(x_training,
                  y_training,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=0,
                  shuffle=True)
    epoch_end_time = time.time()

    results = model.evaluate(x_test, y_test, verbose=0)
    
    y_pred=model.predict(x_test)
    mae=np.sum(np.absolute(y_test[0]-y_pred[0]))*100/np.sum(y_test[0])
    
    for j in range(4, 8):
        weights = model.get_weights()[2*j]
        biases = model.get_weights()[2*j+1]
        np.savetxt('Test_output/'+str(n_bus)+'/W_p_'+str(j-4)+'.csv',weights, fmt='%s', delimiter=',')
        np.savetxt('Test_output/'+str(n_bus)+'/b_p_'+str(j-4)+'.csv',biases, fmt='%s', delimiter=',')
                                
    print("w1",lw1,"w2",lw2,"test loss", mae)
    print("test loss", mae)
    print(results)
    total_end_time = time.time()
    print(f'Total training time: {total_end_time - total_start_time:.1f} seconds')
    # tf.saved_model.save(model, 'Test_output/')

if __name__ == "__main__":
    run_system_identification()