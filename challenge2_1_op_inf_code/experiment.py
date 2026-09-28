from OpInf_utils_Farcas import *

import numpy as np
import h5py as h5
from time import time
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# ============================================================
# SETTINGS
# ============================================================

all_r_start = time()

# timing whole operation
main_start = time()

# DoF setup
ns 	= 2             # u_x and u_y components
n 	= 51*154*2      # number of values per snapshot
nx  = int(n/ns)

# full length of dataset is 12800 snapshots
# nt	= 70
# # number of time instants over the time domain of interest (training + prediction)
# # we are going to predict 100 snapshots from the first 400
# nt_p = 100

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


# Centering and scaling
CENTERING = True
SCALING = True

min_r = 10
max_r = 200

r_vals = [
    r for r in range(min_r, max_r + 1, 10)
]

# Output file
out_file = open(
    "POD_test_reconstruction_50_50.txt",
    "w"
)

# LOAD DATA

print("Loading data...")

with h5.File(H5_training_snapshots, 'r') as file:

    train_ux = file[state_variables[0]][:]
    train_uy = file[state_variables[1]][:]


# Number of available snapshots
total_snapshots = train_ux.shape[0]

print(
    "Total number of snapshots:",
    total_snapshots
)

print(
    "Total number of snapshots: "
    + str(total_snapshots),
    file=out_file
)


# ============================================================
# 50/50 TRAINING / TESTING SPLIT
# ============================================================

nt_train = total_snapshots // 2
nt_test = total_snapshots - nt_train


print(
    "Training snapshots:",
    nt_train
)

print(
    "Testing snapshots:",
    nt_test
)

print(
    "Training snapshots: "
    + str(nt_train),
    file=out_file
)

print(
    "Testing snapshots: "
    + str(nt_test),
    file=out_file
)


# CONSTRUCT TRAINING SNAPSHOT MATRIX

Q_train_original = np.zeros(
    (n, nt_train)
)


# ux
Q_train_original[
    :nx,
    :
] = np.transpose(
    train_ux[:nt_train].reshape(
        nt_train,
        -1
    )
)


# uy
Q_train_original[
    nx:,
    :
] = np.transpose(
    train_uy[:nt_train].reshape(
        nt_train,
        -1
    )
)


# CONSTRUCT TESTING SNAPSHOT MATRIX

Q_test_original = np.zeros(
    (n, nt_test)
)


# ux
Q_test_original[
    :nx,
    :
] = np.transpose(
    train_ux[nt_train:].reshape(
        nt_test,
        -1
    )
)


# uy
Q_test_original[
    nx:,
    :
] = np.transpose(
    train_uy[nt_train:].reshape(
        nt_test,
        -1
    )
)


print(
    "Q_train shape:",
    Q_train_original.shape
)

print(
    "Q_test shape:",
    Q_test_original.shape
)

print(
    "Q_train shape: "
    + str(Q_train_original.shape),
    file=out_file
)

print(
    "Q_test shape: "
    + str(Q_test_original.shape),
    file=out_file
)


# LOOP

results = []


for r in r_vals:

    start = time()

    print("\n")
    print("=" * 60)
    print("POD rank r =", r)
    print("=" * 60)

    print(
        "\n"
        + "=" * 60,
        file=out_file
    )

    print(
        "POD rank r = " + str(r),
        file=out_file
    )

    print(
        "=" * 60,
        file=out_file
    )


    # COPY TRAINING DATA

    Q_global = Q_train_original.copy()


    # SAVE ORIGINAL TRAINING DATA

    Q_global_original = Q_global.copy()


    # CALCULATE TRAINING MEAN
    #
    # The test data will use this same mean.

    temporal_mean_global = np.zeros(n)

    if CENTERING:

        temporal_mean_global = np.mean(
            Q_global,
            axis=1
        )

        Q_global -= (
            temporal_mean_global[:, np.newaxis]
        )


    # CALCULATE TRAINING SCALING
    #
    # The test data will use these same scaling parameters.

    scaling_params = np.ones(ns)


    if SCALING:

        for j in range(ns):

            min_centered_var_global = np.min(
                Q_global[
                    j*nx:(j+1)*nx,
                    :
                ]
            )

            max_centered_var_global = np.max(
                Q_global[
                    j*nx:(j+1)*nx,
                    :
                ]
            )

            scaling_param_global = np.maximum(
                np.abs(
                    min_centered_var_global
                ),
                np.abs(
                    max_centered_var_global
                )
            )

            Q_global[
                j*nx:(j+1)*nx,
                :
            ] /= scaling_param_global

            scaling_params[j] = (
                scaling_param_global
            )


    # APPLY TRAINING TRANSFORMATION TO TEST DATA

    Q_test = Q_test_original.copy()


    if CENTERING:

        Q_test -= (
            temporal_mean_global[:, np.newaxis]
        )


    if SCALING:

        for j in range(ns):

            Q_test[
                j*nx:(j+1)*nx,
                :
            ] /= scaling_params[j]


    # COMPUTE POD BASIS
    #
    # ONLY TRAINING DATA IS USED HERE

    D_global = np.matmul(
        Q_global.T,
        Q_global
    )


    # Eigendecomposition of Gram matrix
    eigs, eigv = np.linalg.eigh(
        D_global
    )


    # Sort eigenvalues from largest to smallest
    sorted_indices = np.argsort(
        eigs
    )[::-1]

    eigs = eigs[
        sorted_indices
    ]

    eigv = eigv[
        :,
        sorted_indices
    ]


    # RETAINED ENERGY

    ret_energy = (
        np.cumsum(eigs)
        /
        np.sum(eigs)
    )


    print(
        "Retained energy:",
        ret_energy[r - 1]
    )

    print(
        "Retained energy at r = "
        + str(r)
        + ": "
        + str(ret_energy[r - 1]),
        file=out_file
    )


    # CONSTRUCT POD TRANSFORMATION

    Tr_global = np.matmul(
        eigv[:, :r],
        np.diag(
            eigs[:r] ** (-0.5)
        )
    )


    # CONSTRUCT POD BASIS
    #
    # Phi_r = Q_train * Tr

    Phir_global = np.matmul(
        Q_global,
        Tr_global
    )


    print(
        "POD basis shape:",
        Phir_global.shape
    )

    print(
        "POD basis shape: "
        + str(Phir_global.shape),
        file=out_file
    )


    # EXPERIMENT:
    #
    # REPRESENT THE TEST DATA USING THE TRAINING POD BASIS

    print(
        "\nProjecting test data onto POD basis..."
    )

    print(
        "\n--- POD TEST REPRESENTATION ---",
        file=out_file
    )


    # Find reduced coordinates of TEST data

    Qhat_test = np.matmul(
        Phir_global.T,
        Q_test
    )


    print(
        "Qhat_test shape:",
        Qhat_test.shape
    )

    print(
        "Qhat_test shape: "
        + str(Qhat_test.shape),
        file=out_file
    )


    # Reconstruct test data from POD basis

    Q_test_POD = np.matmul(
        Phir_global,
        Qhat_test
    )


    # UNDO SCALING
    if SCALING:

        for j in range(ns):

            Q_test_POD[
                j*nx:(j+1)*nx,
                :
            ] *= scaling_params[j]


    # UNDO CENTERING

    if CENTERING:

        Q_test_POD += (
            temporal_mean_global[:, np.newaxis]
        )


    # CALCULATE TEST POD ERROR

    pod_test_error = (
        np.linalg.norm(
            Q_test_original
            -
            Q_test_POD
        )
        /
        np.linalg.norm(
            Q_test_original
        )
    )


    # CALCULATE TRAINING POD ERROR FOR COMPARISON

    Qhat_train = np.matmul(
        Phir_global.T,
        Q_global
    )


    Q_train_POD = np.matmul(
        Phir_global,
        Qhat_train
    )


    # Undo scaling
    if SCALING:

        for j in range(ns):

            Q_train_POD[
                j*nx:(j+1)*nx,
                :
            ] *= scaling_params[j]


    # Undo centering
    if CENTERING:

        Q_train_POD += (
            temporal_mean_global[:, np.newaxis]
        )


    pod_train_error = (
        np.linalg.norm(
            Q_train_original
            -
            Q_train_POD
        )
        /
        np.linalg.norm(
            Q_train_original
        )
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\nPOD training reconstruction error:",
        pod_train_error
    )

    print(
        "POD test reconstruction error:",
        pod_test_error
    )

    print(
        "POD training reconstruction error: "
        + str(pod_train_error),
        file=out_file
    )

    print(
        "POD test reconstruction error: "
        + str(pod_test_error),
        file=out_file
    )


    # STORE RESULTS

    end = time()

    results.append(
        {
            "r": r,
            "retained_energy": ret_energy[r - 1],
            "train_error": pod_train_error,
            "test_error": pod_test_error,
            "time": round(
                end - start,
                5
            )
        }
    )


# SUMMARY TABLE

table = PrettyTable()

table.field_names = [
    "r",
    "Retained Energy",
    "POD Train Error",
    "POD Test Error",
    "Time (sec)"
]


for result in results:

    table.add_row(
        [
            result["r"],
            result["retained_energy"],
            result["train_error"],
            result["test_error"],
            result["time"]
        ]
    )


print("\n")
print(table)


out_file.write(
    "\n\nFINAL RESULTS\n"
)

out_file.write(
    table.get_string()
)


# PLOT POD TEST ERROR VS R

r_plot = [
    result["r"]
    for result in results
]

train_error_plot = [
    result["train_error"]
    for result in results
]

test_error_plot = [
    result["test_error"]
    for result in results
]


plt.figure()

plt.plot(
    r_plot,
    train_error_plot,
    "o-",
    label="Training POD Error"
)

plt.plot(
    r_plot,
    test_error_plot,
    "o-",
    label="Testing POD Error"
)

plt.xlabel("POD Rank r")

plt.ylabel(
    "Relative Reconstruction Error"
)

plt.title(
    "POD Reconstruction Error"
)

plt.legend()

plt.grid()

plt.show()


# SINGULAR VALUE PLOT

# Use the eigenvalues from the final POD calculation

sing_vals = [
    np.sqrt(eigenvalue)
    for eigenvalue in eigs
    if eigenvalue >= 0
]


plt.figure()

plt.semilogy(
    sing_vals,
    "o"
)

plt.xlabel(
    "Index"
)

plt.ylabel(
    "Singular Value"
)

plt.title(
    "Singular Values of Training Snapshot Matrix"
)

plt.show()


# FINISH

all_end = time()

total_time = round(
    all_end - all_r_start,
    5
)

print(
    "\nTotal elapsed time:",
    total_time,
    "seconds"
)

out_file.write(
    "\n\nTotal elapsed time: "
    + str(total_time)
    + " seconds\n"
)

out_file.close()