import FUNCTIONS as fn
import time
from MESH import generate_mesh
from OPTIMIZER import optimize
from RESULT_INTERPRETER import interpret_results, make_VTK
import numpy as np
import os
import csv
import pandas as pd
from OPTIMIZER import optimize_SOCP, optimize_su_solve
from Random_field_Ivan import random_filed

##RUN##
def run():
    #Start timer
    t0 = time.time()

    
    ##Monte Carlo##
    #input_filename = r'MONTE_CARLO\Feasibility.json'
    input_filename_2 = r'MONTE_CARLO\Body_optimize.json'
    input_filename_1 = r'MONTE_CARLO\Su_optimize.json'

    #Loading input
    config_1 = fn.load_config(input_filename_1) #Loading the input parameters
    config_2 = fn.load_config(input_filename_2) #Loading the input parameters

    #
    x, dual_constraints, mesh_info = None, None, (None, None, None, [])

    #Looping for AMR
    for iteration in range(config_1['mesh']['amr_repetitions']):
        #Mesh
        nodes_1, boundaries_1, elements_1, duplicates_1 = generate_mesh(config_1['mesh'], iteration, dual_constraints, mesh_info, x)
        nodes_2, boundaries_2, elements_2, duplicates_2 = generate_mesh(config_2['mesh'], iteration, dual_constraints, mesh_info, x)

        #Constraints 1
        A_1, b_1, node_indicies_1, nodes_to_load_1, nodes_to_skip_on_corner_1, nodes_to_keep_1, body_forces_1, flat_corner_nodes_1, depths_nods_1 = fn.assembly_of_equality_constraints(nodes_1, elements_1, boundaries_1, duplicates_1, config_1['param'], config_1['mesh']) #Generation of equality constraints
        A_ineq_1, b_ineq_1 = fn.A_inequality(config_1['param'], len(nodes_1)) #Generation of inequality constraints
        c_1, optimal_stress_nodes_1 = fn.objective_function(boundaries_1, config_1['mesh']['loaded_lines'], node_indicies_1, duplicates_1, elements_1, nodes_1, config_1['param'], nodes_to_load_1, nodes_to_skip_on_corner_1, nodes_to_keep_1, flat_corner_nodes_1) #Generation of objective function

        #Constraints 2
        A_2, b_2, node_indicies_2, nodes_to_load_2, nodes_to_skip_on_corner_2, nodes_to_keep_2, body_forces_2, flat_corner_nodes_2, depths_nods_2 = fn.assembly_of_equality_constraints(nodes_2, elements_2, boundaries_2, duplicates_2, config_2['param'], config_2['mesh']) #Generation of equality constraints
        A_ineq_2, b_ineq_2 = fn.A_inequality(config_2['param'], len(nodes_2)) #Generation of inequality constraints
        c_2, optimal_stress_nodes_2 = fn.objective_function(boundaries_2, config_2['mesh']['loaded_lines'], node_indicies_2, duplicates_2, elements_2, nodes_2, config_2['param'], nodes_to_load_2, nodes_to_skip_on_corner_2, nodes_to_keep_2, flat_corner_nodes_2) #Generation of objective function

        duplicates_new_copy = duplicates_1.copy()
        duplicates_new_copy2 = duplicates_2.copy()

        ##Monte Carlo##
        for node_ID in nodes_1.keys():
            # Check if the node is already a key in duplicates
            if node_ID in duplicates_1.keys():
                continue  # If node is already a key, move to the next node
            # If not a key, check if the node is in any of the lists of values in duplicates
            found = False
            for value_list in duplicates_1.values():
                if node_ID in value_list:
                    found = True
                    break  # If found as an item, no need to add as a key
            # If the node was not found in any key or list, add it as a key with an empty list as value
            if not found:
                duplicates_1[node_ID] = []

        for node_ID in nodes_2.keys():
            # Check if the node is already a key in duplicates
            if node_ID in duplicates_2.keys():
                continue  # If node is already a key, move to the next node
            # If not a key, check if the node is in any of the lists of values in duplicates
            found = False
            for value_list in duplicates_2.values():
                if node_ID in value_list:
                    found = True
                    break  # If found as an item, no need to add as a key
            # If the node was not found in any key or list, add it as a key with an empty list as value
            if not found:
                duplicates_2[node_ID] = []

        Su_mean = 25.0
        std_dev_Su = 5.0
        N_samp = 100

        coordinates = []
        depths = []

        for uniqe_node in duplicates_1.keys():
            x_coor, y_coor = nodes_1[uniqe_node][0], nodes_1[uniqe_node][1]
            depth_of_t_node = depths_nods_1[uniqe_node]

            coordinates.append([x_coor, y_coor])
            depths.append(depth_of_t_node)

        coordinates = np.array(coordinates)
        depths = np.array(depths).reshape(-1, 1)

        random_field_samples = random_filed(coordinates, depths, Su_mean, std_dev_Su, N_samp)

        #Just an example for now, need to generate sets actually

        fos_su_solve_list = []
        fos_body_solve_list = []
        mean_su_values_from_fields = []
        #Optimalization
        for i in range(random_field_samples.shape[1]):
            print(len(elements_1))
            Su_sample = random_field_samples[:, i]
            #Su_sample = np.random.normal(Su_mean, std_dev_Su, len(duplicates_1))
            #Su_sample = np.ones(len(duplicates_1))*Su_mean
            mes_1, fos_SU, xopt_1, dual_constraints_1, info_1 = optimize_su_solve(c_1, A_1, b_1, config_1['param'], nodes_1, duplicates_1, node_indicies_1, Su_sample)
            mes_2, fos_BODY, xopt_2, dual_constraints_2, info_2 = optimize_SOCP(c_2, A_2, b_2, config_2['param'], body_forces_2, node_indicies_2, duplicates_2, Su_sample)
            
            if info_1['exitFlag'] != 0 or info_2['exitFlag'] != 0:
                continue

            fos_su_solve_list.append(fos_SU)
            fos_body_solve_list.append(fos_BODY)
            mean_su_values_from_fields.append(np.mean(Su_sample))
        
            df_fos = pd.DataFrame({
                'FOS_Su_Solve': fos_su_solve_list,
                'FOS_Body_Solve': fos_body_solve_list,
                'Mean_Su': mean_su_values_from_fields
            })
            df_fos.to_csv('tests.csv', index=False)

        #df_fos.to_csv('tests.csv', index=False)

        make_VTK(xopt_1, nodes_1, elements_1, node_indicies_1, duplicates_new_copy, 'Optimized_su', dual_constraints_1, config_1['param'])
        make_VTK(xopt_2, nodes_2, elements_2, node_indicies_2, duplicates_new_copy2, 'Optimized_body', dual_constraints_2, config_2['param'])

        '''
        opt_value = FOS_su_solve
        if x is None:
            print("Optimization failed, terminating program.")
            return  # Exit the function early
        stress = interpret_results(config['results'], status, opt_value, optimal_stress_nodes, x, c, nodes, node_indicies, elements, duplicates, A, b, config['param'], boundaries, config['mesh']['loaded_lines'], config['mesh'], dual_constraints)

        ###SAVING DATA TO CSV###
        total_time = time.time()-t0
        print(f'Total execution time: {total_time:.2f} seconds')
        headers = ['Simulation_name', 'NE', 'q', 'solver_time', 'run_time', 'total_time', 'solver', 'iter', 'pres', 'dres', 'gap', 'exitFlag']
        data = [config['results']['save_VTK_as'], len(elements), -stress, inr['timing']['tsolve'], inr['timing']['runtime'], total_time, config['param']['solver'], inr['iter'], inr['pres'], inr['dres'], inr['gap'], inr['exitFlag']]
        file_path = 'C:\\Users\\danie\\OneDrive - NTNU\\Documents\\Ntnu\\2024 vår\\Master\\Final results\\Case 3 - Slope Stability\\results_UNIFORM.csv'
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(data)
        ##########################


        # Check if we need to continue with the next iteration of AMR
        if not config['mesh']['amr'] or iteration >= config['mesh']['amr_repetitions']-1:
            break

        print(f"AMR iteration {iteration}/{config['mesh']['amr_repetitions']-1} complete.")

        mesh_info = [node_indicies, nodes, elements, duplicates, config['param']]
    
    #if config['results']['make_VTK']:
    #    make_VTK(x, nodes, elements, node_indicies, duplicates, config['results']['save_VTK_as'], dual_constraints)
'''

if __name__ == '__main__':
    run()