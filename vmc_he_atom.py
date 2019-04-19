import numpy as np
import matplotlib.pyplot as plt


def energy_local(r1, r2, alpha):

    r1_unit_vec = np.multiply(r1.T, 1/np.linalg.norm(r1, axis=1)).T
    r2_unit_vec = np.multiply(r2.T, 1/np.linalg.norm(r2, axis=1)).T
    r12 = r1 - r2
    r12_norm = np.linalg.norm(r12, axis=1)
    return -4 + np.einsum('ij,ij->i', r1_unit_vec - r2_unit_vec, r12)/(r12_norm * (1 + alpha * r12_norm)**2) - \
           1/(r12_norm * (1 + alpha * r12_norm)**3) - 1/(4 * (1 + alpha * r12_norm)**4) + 1/r12_norm


def trans_prob(r1, r2, r1_p, r2_p, alpha):

    r1_norm = np.linalg.norm(r1, axis=1)
    r2_norm = np.linalg.norm(r2, axis=1)
    r1_p_norm = np.linalg.norm(r1_p, axis=1)
    r2_p_norm = np.linalg.norm(r2_p, axis=1)
    r12_norm = np.linalg.norm(r1 - r2, axis=1)
    r12_p_norm = np.linalg.norm(r1_p - r2_p, axis=1)

    psi = np.exp(-2*(r1_norm + r2_norm) + r12_norm/(2*(1 + alpha * r12_norm)))
    psi_p = np.exp(-2*(r1_p_norm + r2_p_norm) + r12_p_norm/(2*(1 + alpha * r12_p_norm)))

    return np.square(psi_p / psi)


def vmc(num_walkers, num_mc_steps, num_therm_steps, alpha, new_walker_std=1.0):

    if num_mc_steps <= num_therm_steps:
        print("ERROR: Number of steps must be greater than number of thermalizing steps")
        return 0, 0

    # Initialize walkers and energy
    walkers1 = np.random.rand(num_walkers, 3)
    walkers2 = np.random.rand(num_walkers, 3)
    energy_estimator = np.zeros([num_mc_steps-num_therm_steps, num_walkers])
    num_accept_jumps = 0

    for step_MC in range(num_mc_steps):

        progress = 100*step_MC/num_mc_steps
        if progress % 5 == 0:
            print("Progress: " + str(progress) + "%")

        # Randomly create new walkers
        new_walkers1 = np.random.normal(walkers1, scale=new_walker_std)
        new_walkers2 = np.random.normal(walkers2, scale=new_walker_std)


        # Test new walkers and update
        update_conditions = np.random.rand(num_walkers) < trans_prob(walkers1, walkers2, new_walkers1, new_walkers2, alpha)
        walkers1[update_conditions,:] = new_walkers1[update_conditions,:]
        walkers2[update_conditions,:] = new_walkers2[update_conditions,:]
        num_accept_jumps += np.sum(update_conditions)

        # Evaluate local energy at walkers positions
        if step_MC >= num_therm_steps:
            energy_estimator[step_MC - num_therm_steps, :] = energy_local(walkers1, walkers2, alpha)

    accept_ratio = num_accept_jumps / (num_walkers*num_mc_steps)

    return energy_estimator, accept_ratio


if __name__ == "__main__":
    num_walk = 500
    num_steps = 20000
    therm_steps = 5000
    alphas = np.linspace(0.05, 0.25, num=50)
    energies = np.zeros(len(alphas))
    energies_std = np.zeros(len(alphas))
    accept_rats = np.zeros(len(alphas))
    i = 0

    for a in alphas:
        print("alpha = " + str(a))

        energy_est, accept_rat = vmc(num_walk, num_steps, therm_steps, alpha=a)
        if np.isscalar(energy_est):
            break
        energies[i] = np.average(energy_est.flatten())
        energies_std[i] = np.std(energy_est.flatten())
        accept_rats[i] = accept_rat
        i += 1

    print(energies)
    print(accept_rats)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(alphas, energies, yerr=energies_std, ecolor="black", elinewidth=1.0, capsize=3,
                marker="*", color="red", linestyle="--", linewidth=1.0, label="VMC")
    ax.legend()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Energy estimation")
    plt.show()