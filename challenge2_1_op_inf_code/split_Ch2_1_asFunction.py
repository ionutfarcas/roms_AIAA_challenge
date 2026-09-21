from OpInf_utils_Farcas import *

import numpy as np
import h5py as h5
from itertools import product
from time import time
import matplotlib.pyplot as plt
from prettytable import PrettyTable

all_r_start = time()

# timing whole operation
main_start = time()

# DoF setup
ns 	= 2             # u_x and u_y components
n 	= 51*154*2      # number of values per snapshot
nx  = int(n/ns)

# full length of dataset is 12800 snapshots
nt	= 70
# number of time instants over the time domain of interest (training + prediction)
# we are going to predict 100 snapshots from the first 400
nt_p = 100

# state variable names
state_variables = ['ux', 'uy']

# path to the HDF5 file containing the training snapshots
# this was downloaded from the google drive that the challenge shared
H5_training_snapshots 	= '../challenge_data/Challenge2_1_train.h5' 

# define target retained energy for the OpInf ROM
target_ret_energy = 0.9996

# ranges for the regularization parameter pairs
B1 = np.logspace(-8., 8., num=50)
# B1 = np.logspace(-10., 10., num=50) # the bigger space was for testing on a stronger computer than my laptop

B2 = np.logspace(-8., 8., num=50)
# B2 = np.logspace(-10., 10., num=50) # or 100


# maximum variance of reduced training data for optimal regularization parameter selection
max_growth = 1.2

# flag to determine whether the (transformed) training data are centered with respect to the temporal mean 
CENTERING = True

# flag to determine whether the (transformed) training data are scaled by the maximum absolute value of each state variable
SCALING = True

# flag to determine whether we postprocess the OpInf reduced solution
POSTPROC = True

# flag to determine whether we compute the ROM approximate solution in the original cooridnates in the full domain 
# at user specified time instants (specified in target_time_instants)
POSTPROC_FULL_DOM_SOL = True

# flag to determine whether we compute the ROM approximate solution in the original coordinates
# at user specified probe locations (specified in target_probe_indices)
POSTPROC_PROBES = True

# time instants at which to save the OpInf approximate solutions mappend to the original coordinates
target_time_instants 	= [-1]

# setting up output file
out_file  = open("train_output_nt" + str(nt) +".txt", "w")

def solve(r):
	######## INITIALIZATION ########
	# compute the Cartesian product of all regularization pairs (beta1, beta2 )
	reg_pairs_global 	= list(product(B1, B2))
	n_reg_global 		= len(reg_pairs_global)
	###### INITIALIZATION END ######

	######## STEP I: SEQUENTIAL TRAINING DATA LOADING ########
	# allocate memory for the global snapshot data, which has been saved to disk in HDF5 format
	load_t = time()
	Q_global = np.zeros((n, nt))
	with h5.File(H5_training_snapshots, 'r') as file:
		train_ux = file[state_variables[0]][:]
		train_uy = file[state_variables[1]][:]
	file.close()

	load_t2 = time()
	
    # Recording total number of snapshots at our disposal
	out_file.write("Total number of x components: " + str(train_ux.shape[0]))
	out_file.write("\nTotal number of y components: " + str(train_uy.shape[0]))

	# taking the first nt snapshots 
	curr_ux = train_ux[:nt]
	curr_uy = train_uy[:nt]

	# reshaping each snapshot to a single vector
	j = 0
	Q_global[j*nx : (j + 1)*nx, :] = np.transpose(curr_ux.reshape(curr_ux.shape[0], -1))
	j = 1
	Q_global[j*nx : (j + 1)*nx, :] = np.transpose(curr_uy.reshape(curr_uy.shape[0], -1))

	print("\nQ global shape: " + str(Q_global.shape), file=out_file)

	# taking the testing snapshots 
	ux_p = train_ux[nt: nt_p]
	uy_p = train_uy[nt: nt_p]

	# saving the snapshots that correspond to the testing snapshots
	Q_global_pred = np.zeros((n, nt_p - nt))
	j = 0
	Q_global_pred[j*nx : (j + 1)*nx, :] = np.transpose(ux_p.reshape(ux_p.shape[0], -1))
	j = 1
	Q_global_pred[j*nx : (j + 1)*nx, :] = np.transpose(uy_p.reshape(uy_p.shape[0], -1))


	#################### STEP I END ###########################
	

	######## STEP II: SEQUENTIAL DATA TRANSFORMATIONS ########
	## Finding min and max for each variable to debate scaling
	j=0
	print("\n\nx component: min = " + str(np.min(Q_global[j*nx : (j + 1)*nx, :])) + "   max = "+ str(np.max(Q_global[j*nx : (j + 1)*nx, :])), file=out_file)
	j=1
	print("y component: min = " + str(np.min(Q_global[j*nx : (j + 1)*nx, :])) + "   max = "+ str(np.max(Q_global[j*nx : (j + 1)*nx, :])), file=out_file)
	Q_global_original = Q_global.copy()
	scaling_params = np.ones(ns)
    ## actually performing the desired operations
	if CENTERING:
		# compute the global temporal mean of each variable
		temporal_mean_global 	= np.mean(Q_global, axis=1)
		# center (in place) each variable with respect to its global temporal mean
		Q_global 				-= temporal_mean_global[:, np.newaxis]

	if SCALING:
        # scale the centered stated variables by their global maximum absolute value
		# this ensures that the centered and scaled variables do not exceed [-1, 1]
		for j in range(ns):
            # determine the global maximum absolute value of each centered variable on each rank
			min_centered_var_global = np.min(Q_global[j*nx : (j + 1)*nx, :])
			max_centered_var_global = np.max(Q_global[j*nx : (j + 1)*nx, :])
			scaling_param_global = np.maximum(np.abs(min_centered_var_global), \
			                                     np.abs(max_centered_var_global))

			# scale each centered variable by its corresponding global scaling parameter
			Q_global[j*nx : (j + 1)*nx, :] /= scaling_param_global
			scaling_params[j] = np.maximum(np.abs(min_centered_var_global), np.abs(max_centered_var_global))
	#################### STEP II END ##########################

	
	######## STEP III: SEQUENTIAL DIMENSIONALITY REDUCTION ########
	comp_t = time()
	# compute the local Gram matrices on each rank
	D_global  			= np.matmul(Q_global.T, Q_global)
	
	# compute the eigendecomposition of the positive, semi-definite global Gram matrix
	eigs, eigv = np.linalg.eigh(D_global)

	# order eigenpairs by increasing eigenvalue magnitude
	sorted_indices 	= np.argsort(eigs)[::-1]
	eigs 			= eigs[sorted_indices]
	eigv 			= eigv[:, sorted_indices]

	# compute retained energy for r bteween 1 and nt
	ret_energy 	= np.cumsum(eigs)/np.sum(eigs)
	# select reduced dimension r for that the retained energy exceeds the prescribed threshold
	# r 			= np.argmax(ret_energy > target_ret_energy) + 1
	# r = 15

	print("\n\nChose r = ", str(r), file= out_file)

	# compute the auxiliary Tr matrix
	Tr_global 	= np.matmul(eigv[:, :r], np.diag(eigs[:r]**(-0.5)))
	# compute the low-dimensional representation of the high-dimensional transformed snapshot data
	Qhat_global = np.matmul(Tr_global.T, D_global)
	comp_t2 = time()
	##################### STEP III END #############################


	######## STEP IV: SEQUENTIAL REDUCED OPERATOR INFERENCE ########
	# extract left and right shifted reduced data matrices for the discrete OpInf learning problem
	Qhat_1 = Qhat_global.T[:-1, :]
	Qhat_2 = Qhat_global.T[1:, :]

	# column dimension of the data matrix Dhat used in the discrete OpInf learning problem
	s = int(r*(r + 1)/2)
	d = r + s + 1        # at most n_t - 1
	print("\nCorresponding d = ", str(d), file= out_file)

	# compute the non-redundant quadratic terms of Qhat_1 squared
	print("\nQhat_1 shape", Qhat_1.shape, file = out_file)
	Qhat_1_sq = compute_Qhat_sq(Qhat_1)

	# define the constant part (due to mean shifting) in the discrete OpInf learning problem
	K 		= Qhat_1.shape[0]
	Ehat 	= np.ones((K, 1))

	# assemble the data matrix Dhat for the discrete OpInf learning problem
	Dhat   = np.concatenate((Qhat_1, Qhat_1_sq, Ehat), axis=1)
	# compute Dhat.T @ Dhat for the normal equations to solve the OpInf least squares minimization
	Dhat_2 = Dhat.T @ Dhat

	# compute the temporal mean and maximum deviation of the reduced training data
	mean_Qhat_train   	 	= np.mean(Qhat_global.T, axis=0)
	max_diff_Qhat_train 	= np.max(np.abs(Qhat_global.T - mean_Qhat_train), axis=0)
	# training error corresponding to the optimal regularization hyperparameters
	opt_train_err 			= 1e20

	# recording error
	errors = []

	# letting this have a value if no pairs produce an error smaller than the max
	beta1_opt = None

	# loop over the regularization pairs corresponding to each MPI rank
	count = 0
	for pair in reg_pairs_global:

		# extract beta1 and beta2 from each candidate regularization pair
		beta1 = pair[0]
		beta2 = pair[1]

		# regularize the linear and constant reduced operators using beta1, and the quadratic operator using beta2
		regg            = np.zeros(d)
		regg[:r]        = beta1
		regg[r : r + s] = beta2
		regg[r + s:]    = beta1
		regularizer     = np.diag(regg)
		Dhat_2_reg 		= Dhat_2 + regularizer

		# solve the OpInf learning problem by solving the regularized normal equations
		Ohat = np.linalg.solve(Dhat_2_reg, np.dot(Dhat.T, Qhat_2)).T

		# extract the linear, quadratic, and constant reduced model operators
		Ahat = Ohat[:, :r]
		Fhat = Ohat[:, r:r + s]
		chat = Ohat[:, r + s]

		# define the OpInf reduced model 
		dOpInf_red_model 	= lambda x: Ahat @ x + Fhat @ compute_Qhat_sq(x) + chat
		# extract the reduced initial condition from Qhat_1
		qhat0 				= Qhat_1[0, :]
		
		# compute the reduced solution over the trial time horizon, which here is the same as the target time horizon
		contains_nans, Qtilde_OpInf 	= solve_opinf_difference_model(qhat0, nt_p, dOpInf_red_model)
		
		# for each candidate regulairzation pair, we compute the training error 
		# we also save the corresponding reduced solution, learning time and ROM evaluation time
 		# and compute the ratio of maximum coefficient growth in the trial period to that in the training period
		if contains_nans == False:
			train_err     			= compute_train_err(Qhat_global.T[:nt, :], Qtilde_OpInf[:nt, :])
			max_diff_Qhat_trial  	= np.max(np.abs(Qtilde_OpInf - mean_Qhat_train), axis=0)			
			max_growth_trial  		= np.max(max_diff_Qhat_trial)/np.max(max_diff_Qhat_train)
			errors.append(train_err)

			if max_growth_trial < max_growth:
				count = count +1
				if train_err < opt_train_err:
					opt_train_err 				= train_err
					Qtilde_OpInf_opt 			= Qtilde_OpInf

					beta1_opt = pair[0]
					beta2_opt = pair[1]
		else:
			errors.append(np.nan)

	# print("\n\nNumber of regularization pairs that produced a solution with max growth < " + str(max_growth) + " : " + str(count))
	# B1_grid, B2_grid = np.meshgrid(B1, B2, indexing='ij')

	# Z = np.array(errors).reshape(len(B1), len(B2))

	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')

	# surf = ax.plot_surface(
	# 	np.log10(B1_grid),
	# 	np.log10(B2_grid),
	# 	Z,
	# 	cmap='viridis'
	# )

	# ax.set_xlabel('log10(beta1)')
	# ax.set_ylabel('log10(beta2)')
	# ax.set_zlabel('Training Error')

	# fig.colorbar(surf, shrink=0.5, aspect=5)

	# plt.show()
	# ####################### STEP IV END #############################
	print("Optimal training error: ", opt_train_err, file=out_file)
	# reporting chosen beta1 and beta2
	if (beta1_opt != None):
		print("Optimal regularization pair: (" + str(beta1_opt) + ", " + str(beta2_opt) + ")\n", file=out_file)


	if (POSTPROC and (beta1_opt != None)):
		######## POSTPROCESSING ########
		if POSTPROC_FULL_DOM_SOL:
			# compute the global POD basis vectors
			Phir_global = np.matmul(Q_global, Tr_global)
			print("Phi_r Global size: ", np.size(Phir_global))
			Qhat_check = Phir_global.T @ Q_global
			Q_rec_scaled = Phir_global @ Qhat_check
			Q_rec = Q_rec_scaled.copy()

			Q_rec[:nx, :] *= scaling_params[0]
			Q_rec[nx:, :] *= scaling_params[1]

			if CENTERING:
				Q_rec += temporal_mean_global[:, None]			
			pod_error = np.linalg.norm(
				Q_global_original - Q_rec
			) / np.linalg.norm(Q_global_original)

			print("POD reconstruction error:", pod_error)
			# extract and save to disk the approximate full state at the time instants specified in target_time_instants 
			#for target_var_index in range(ns):
			#	Phir_full_state 			= Phir_global[target_var_index*nx : (target_var_index + 1)*nx, :]
			#	temporal_mean_full_state 	= temporal_mean_global[target_var_index*nx : (target_var_index + 1)*nx]

			#	full_state_rec = Phir_full_state @ Qtilde_OpInf_opt.T[:, target_time_instants] + temporal_mean_full_state[:, np.newaxis]
				
			#	np.save('postprocessing/sOpInf_postprocessing/sOpInf_full_state_var_' + str(target_var_index + 1) + '.npy', full_state_rec)
			

			# extracting approximate full states all time instances
			Phir_full_state = Phir_global
			temporal_mean_full_state = temporal_mean_global

			# Reconstruct in centered/scaled coordinates
			full_state_rec = Phir_full_state @ Qtilde_OpInf_opt.T

			# Undo scaling
			if SCALING:
				for j in range(ns):
					full_state_rec[j*nx:(j+1)*nx, :] *= scaling_params[j]

			# Undo centering
			if CENTERING:
				full_state_rec += temporal_mean_full_state[:, np.newaxis]			
			print(Phir_full_state.shape)
			print(full_state_rec.shape)
			print(Q_global.shape)

			# computing error in the original basis for the training data
			full_solution_error_nt = compute_train_err(
				Q_global_original.T,
				full_state_rec.T[:nt, :]
			)

			print("Normalized Error for both state vars:    " + str(full_solution_error_nt), file=out_file)

			# computing the error for the predicted snapshots
			pred_err = compute_train_err(
			Q_global_pred.T,
			full_state_rec.T[nt:, :])			

			print("Normalized Error for both state vars:    " + str(pred_err), file=out_file)


			# computing the error the same way the Challenge writers did
			chal_err_train = (
				np.mean((Q_global_original.T - full_state_rec.T[:nt, :])**2)
				/ np.mean(Q_global_original**2)
			)			
			chal_err_test = (
				np.mean((Q_global_pred.T - full_state_rec.T[nt:, :])**2)
				/ np.mean(Q_global_pred**2)
			)
			print("Training Error defined from Challenge:", chal_err_train)
			print("Forecasting Error defined from Challenge:", chal_err_test)

			print("Training Error defined from Challenge: " + str(chal_err_train), file=out_file)
			print("Forecasting Error defined from Challenge: " + str(chal_err_test), file=out_file)


			# comparing against unused singular values
			sigmas = [np.sqrt(eigenvalue) for eigenvalue in eigs[r:] if eigenvalue >=0 ]
			sigma_error = sum(sigmas)			
			print("Error from unused sigmas:                " + str(sigma_error), file=out_file)


		# extract and save to disk the approximate solutions at the probe locations specified in target_probe_indices
		'''
		for target_var_index in range(ns):

			for j, probe_index in enumerate(target_probe_indices):

				# compute the components of the POD basis vectors corresponding to the target probe locations
				Phir_probe 			= np.matmul(Q_global[probe_index + target_var_index*nx, :], Tr_global)
				print("Phir_probe", Phir_probe.shape)
				temporal_mean_probe = temporal_mean_global[probe_index + target_var_index*nx]

				var_probe_prediction = Phir_probe @ Qtilde_OpInf_opt.T + temporal_mean_probe
				print("var_probe_prediction", var_probe_prediction.shape)


				#np.save('postprocessing/ch_2_1_post/sOpInf_probe_' + str(j + 1) + '_var_' + str(target_var_index + 1) + '.npy', var_probe_prediction)
		'''
		end_time_postproc = time()

	# the case when the error of any canidate point never went below the maximum
	else:
		print("Beta 1 is NONE")
		full_solution_error_nt = None
		pred_err = None
		# comparing against unused sigmas
		sigmas = [np.sqrt(eigenvalue) for eigenvalue in eigs[r:] if eigenvalue >=0]
		sigma_error = sum(sigmas)
		print("Error from unused sigmas:                " + str(sigma_error), file=out_file)
		###### POSTPROCESSING END ######
	main_end = time()
	# reporting times
	print("= "*30, file=out_file)
	print("Data Load time:        ", round(load_t2 - load_t, 5), " seconds ", file= out_file)
	print("Computing Qhat Global: ", round(comp_t2 - comp_t, 5), " seconds ", file= out_file)
	print("Total runtime:         ", round(main_end - main_start, 5), " seconds ", file= out_file)


	return [opt_train_err, full_solution_error_nt, pred_err, sigma_error, eigs, round(main_end - main_start, 5)]

	

# looping over rank values for the given nt 
training = []
full = []
pred = []
sigma = []
eigs_nested = []
times = []
min_r = 20
max_r = 50

# possible max rank values
r_vals = [r for r in range(min_r, max_r+1, 10)]

for r in r_vals:
	print(r)
	train_err, full_err_nt, pred_err, sigma_err, eigs, elapsed_t = solve(r)
	training.append(train_err)
	full.append(full_err_nt)
	pred.append(pred_err)
	sigma.append(sigma_err)
	eigs_nested.append(eigs)
	times.append(elapsed_t)
	out_file.write("\n" + "**"*30 + "\n\n")


# organizing
printed_tbl = PrettyTable()
printed_tbl.field_names = ["Training Error", "Normalized Error for x & y", "Error on prediction snapshots", 
						   "sum(unused sigma)", "time (sec)"]
for i in range(len(training)):
	printed_tbl.add_row([training[i], full[i], pred[i], sigma[i], times[i]])
row_name = "r"
printed_tbl.field_names.insert(0, row_name)
printed_tbl._align[row_name] = 'l' 
printed_tbl._valign[row_name] = 't' 
for i in range(len(training)):
    printed_tbl.rows[i].insert(0, r_vals[i])

out_file.write("\n\n\nError table\n")
out_file.write(printed_tbl.get_string())


# finished all loop iterations
all_r_end = time()

print("\n\n\nTotal Elapsed Time: " + str(round(all_r_end - all_r_start, 5)) + " seconds", file= out_file)

out_file.close()

figure = plt.figure()
sing_vals = [np.sqrt(eigen_v) for eigen_v in eigs if eigen_v >=0]	# computer error sometimes gives negative ones	
plt.semilogy(sing_vals, "o")
plt.ylabel("Log(Eigenalues)")
plt.title("Singular Values of Q by index")
plt.show()


