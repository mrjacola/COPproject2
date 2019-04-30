import numpy as np
import matplotlib.pyplot as plt

# This program calculates ground state energy for the He-atom using Variational Monte Carlo
# In this version Stochastic Gradient Approximation (SGA) is used to find optimal parameters


def energy_local(r, alpha):
    r1_unit_vec = np.multiply(r[:, :, 0].T, 1/np.linalg.norm(r[:, :, 0], axis=1)).T
    r2_unit_vec = np.multiply(r[:, :, 1].T, 1/np.linalg.norm(r[:, :, 1], axis=1)).T
    r12 = r[:,:,0] - r[:,:,1]
    r12_norm = np.linalg.norm(r12, axis=1)

    return -4 + np.einsum('ij,ij->i', r1_unit_vec - r2_unit_vec, r12)/(r12_norm * (1 + alpha * r12_norm)**2) - \
            1/(r12_norm * (1 + alpha * r12_norm)**3) - 1/(4 * (1 + alpha * r12_norm)**4) + 1/r12_norm


def trans_prob(r, r_p, alpha):
    r1_norm = np.linalg.norm(r[:, :, 0], axis=1)
    r2_norm = np.linalg.norm(r[:, :, 1], axis=1)
    r1_p_norm = np.linalg.norm(r_p[:, :, 0], axis=1)
    r2_p_norm = np.linalg.norm(r_p[:, :, 1], axis=1)
    r12_norm = np.linalg.norm(r[:, :, 0] - r[:, :, 1], axis=1)
    r12_p_norm = np.linalg.norm(r_p[:, :, 0] - r_p[:, :, 1], axis=1)

    psi = np.exp(-2*(r1_norm + r2_norm) + r12_norm/(2*(1 + alpha * r12_norm)))
    psi_p = np.exp(-2*(r1_p_norm + r2_p_norm) + r12_p_norm/(2*(1 + alpha * r12_p_norm)))

    return np.square(psi_p / psi)


def energy_grad(r, alpha):
    # Calc gradient with respect to parameter alpha using SGA
    energy_l = energy_local(r, alpha)
    energy = np.average(energy_l)
    r12_norm = np.linalg.norm(r[:, :, 0] - r[:, :, 1], axis=1)
    log_psi_grad = - 0.5 * np.square(r12_norm / (1 + alpha * r12_norm))

    return 2 * (np.average(energy_l * log_psi_grad) - energy * np.average(log_psi_grad))


def step_size(i):
    # Specifies step-dependent step size used for gradient descent
    scale = 0.5
    exponent = - 0.8
    return scale * np.power(i + 1.0, exponent)


def vmc_sga(num_walkers, num_mc_steps, num_thermalizing_steps, alpha_0=0.13, new_walker_std=1.0):
    # Performs the MC calculations

    # Init
    counted_steps = num_mc_steps - num_thermalizing_steps
    walkers = np.random.rand(num_walkers, 3, 2)
    energy_estimator = np.zeros([counted_steps])
    alpha_estimator = np.zeros([counted_steps])
    alpha = alpha_0
    num_accept_jumps = 0

    for step_mc in range(num_mc_steps):

        progress = 100 * step_mc / num_mc_steps
        if progress % 5 == 0:
            print("Progress: " + str(progress) + "%")

        # Randomly create new walkers
        new_walkers = np.random.normal(walkers, scale=new_walker_std)

        # Test new walkers and update new positions
        update_conditions = np.random.rand(num_walkers) < trans_prob(walkers, new_walkers, alpha)
        walkers[update_conditions, :, :] = new_walkers[update_conditions, :, :]
        num_accept_jumps += np.sum(update_conditions)

        # Evaluate energy at walkers positions and update alpha
        if step_mc >= num_thermalizing_steps:
            actual_step = step_mc - num_therm_steps
            energy_estimator[actual_step] = np.average(energy_local(walkers, alpha))
            alpha = alpha - step_size(actual_step) * energy_grad(walkers, alpha)
            alpha_estimator[actual_step] = alpha

    accept_ratio = num_accept_jumps / (num_walkers * num_mc_steps)

    return energy_estimator, alpha_estimator, accept_ratio


if __name__ == "__main__":
    # Example on running calculations

    # Parameters
    num_walk = 1000
    num_steps = 40000
    num_therm_steps = 5000

    # Run
    energy_est, alpha_est, accept_rat = vmc_sga(num_walk, num_steps, num_therm_steps)

    print("Final energy: " + str(energy_est[-1]))
    print("Final alpha: " + str(alpha_est[-1]))
    print("Total accept ratio: " + str(accept_rat))

    fig_alpha = plt.figure(num=1)
    ax1 = fig_alpha.add_subplot(111)
    ax1.set_ylabel(r"$\alpha$")
    ax1.set_xlabel("Iteration")
    ax1.plot(alpha_est, "k", linewidth=1.0)

    fig_energy = plt.figure(num=2)
    ax2 = fig_energy.add_subplot(111)
    ax2.set_ylabel("Energy")
    ax2.set_xlabel("Iteration")
    ax2.plot(energy_est, "r", linewidth=1.0)

    fig_e_a = plt.figure(num=3)
    ax3 = fig_e_a.add_subplot(111)
    ax3.set_ylabel("Energy")
    ax3.set_xlabel(r"$\alpha$")
    ax3.scatter(alpha_est, energy_est, s=0.5)

    plt.show()
