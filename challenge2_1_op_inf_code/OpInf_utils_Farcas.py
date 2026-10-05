import numpy as np

def distribute_nx(rank, nx, size):
	"""
 	distribute_nx distributes the spatial DoF nx into chunks of size nxi such that 
 	sum_{i=0}^{p-1} nx_i = nx where p is the number of used compute cores

 	:rank: 	the MPI rank 0, 1, ... ,p-1 that will run this function
 	:n_x: 	number of DoF used for spatial discretization
 	:size: 	size of the MPI communicator (p in our case)
 	
 	:return: the start and end index of the local DoF, and the number of local DoF for each rank
 	"""

	nx_i_equal = int(nx/size)

	nx_i_start = rank * nx_i_equal
	nx_i_end   = (rank + 1) * nx_i_equal

	if rank == size - 1 and nx_i_end != nx:
		nx_i_end += nx - size*nx_i_equal

	nx_i = nx_i_end - nx_i_start

	return nx_i_start, nx_i_end, nx_i

def distribute_reg_pairs(rank, n_reg, size):
	"""
 	get_reg_params_per_rank returns the index of the first and last regularization pair for each MPI rank
 
 	:rank: 		MPI rank 0, 1, ... ,p-1 
 	:n_reg: 	total number of regularization parameter pairs
 	:size: 		size of the MPI communicator (p in our case)

 	:return: the start and end indices, and the total number of snapshots for each MPI rank
 	"""

	nreg_i_equal = int(n_reg/size)

	start = rank * nreg_i_equal
	end   = (rank + 1) * nreg_i_equal

	if rank == size - 1 and end != n_reg:
		end += n_reg - size*nreg_i_equal

	return start, end

def compute_Qhat_sq(Qhat):
	"""
	compute_Qhat_sq returns the non-redundant terms in Qhat squared

	:Qhat: reduced data

	:return: Qhat_sq containing the non-redundant in Qhat squared
	"""

	if len(np.shape(Qhat)) == 1:
		r = np.size(Qhat)
		prods 	= []
		for i in range(r):
			temp = Qhat[i]*Qhat[i:]
			prods.append(temp)
		
		Qhat_sq = np.concatenate(tuple(prods))

	elif len(np.shape(Qhat)) == 2:
		K, r = np.shape(Qhat)
		prods = []
	
		for i in range(r):
			temp = np.transpose(np.broadcast_to(Qhat[:, i], (r - i, K)))*Qhat[:, i:]
			prods.append(temp)
	
		Qhat_sq = np.concatenate(tuple(prods), axis=1)
	
	else:
		print('invalid input!')
		
	return Qhat_sq



def compute_train_err(Qhat_train, Qtilde_train):
	"""
	compute_train_err computes the OpInf training error

	:Qhat_train: 	Qhat_trainerence data
	:Qtilde_train: 	Qtilde_train data

	:return: train_err containing the value of the training error
	"""
	train_err = np.max(np.sqrt(np.sum( (Qtilde_train - Qhat_train)**2, axis=1) / np.sum(Qhat_train**2, axis=1)))

	return train_err

def solve_opinf_difference_model(
    qhat0,
    n_steps_pred,
    dOpInf_red_model,
    max_norm=1000.0
):
    """
    Solve the discrete OpInf ROM over n_steps_pred time steps.

    Terminates early if the reduced state becomes non-finite or exceeds
    max_norm. This prevents unstable beta candidates from wasting time
    propagating an already-diverged solution.

    Returns
    -------
    contains_nans : bool
        True if the trajectory became unstable/non-finite or exceeded max_norm.
    Qtilde : ndarray
        Reduced trajectory with shape (n_steps_completed, r).
    """

    r = np.size(qhat0)

    Qtilde = np.zeros((r, n_steps_pred))
    Qtilde[:, 0] = qhat0

    for i in range(n_steps_pred - 1):

        x = Qtilde[:, i]

        # Check current state before evaluating the model
        if not np.all(np.isfinite(x)):
            return True, Qtilde[:, :i + 1].T

        # Stop if solution is clearly diverging
        if np.max(np.abs(x)) > max_norm:
            return True, Qtilde[:, :i + 1].T

        # Evaluate next state
        x_next = dOpInf_red_model(x)

        # Check for NaN/Inf immediately
        if not np.all(np.isfinite(x_next)):
            return True, Qtilde[:, :i + 1].T

        # Stop runaway trajectories
        if np.max(np.abs(x_next)) > max_norm:
            return True, Qtilde[:, :i + 2].T

        Qtilde[:, i + 1] = x_next

    return False, Qtilde.T
