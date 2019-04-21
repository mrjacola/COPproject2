import numpy as np
import matplotlib.pyplot as plt


def energy_local(r, alpha):

    r1_unit_vec = np.multiply(r[:,:,0].T, 1/np.linalg.norm(r[:,:,0], axis=1)).T
    r2_unit_vec = np.multiply(r[:,:,1].T, 1/np.linalg.norm(r[:,:,1], axis=1)).T
    r12 = r[:,:,0] - r[:,:,1]
    r12_norm = np.linalg.norm(r12, axis=1)
    return -4 + np.einsum('ij,ij->i', r1_unit_vec - r2_unit_vec, r12)/(r12_norm * (1 + alpha * r12_norm)**2) - \
           1/(r12_norm * (1 + alpha * r12_norm)**3) - 1/(4 * (1 + alpha * r12_norm)**4) + 1/r12_norm


def trans_prob(r, r_p, alpha):

    r1_norm = np.linalg.norm(r[:,:,0], axis=1)
    r2_norm = np.linalg.norm(r[:,:,1], axis=1)
    r1_p_norm = np.linalg.norm(r_p[:,:,0], axis=1)
    r2_p_norm = np.linalg.norm(r_p[:,:,1], axis=1)
    r12_norm = np.linalg.norm(r[:,:,0] - r[:,:,1], axis=1)
    r12_p_norm = np.linalg.norm(r_p[:,:,0] - r_p[:,:,1], axis=1)

    psi = np.exp(-2*(r1_norm + r2_norm) + r12_norm/(2*(1 + alpha * r12_norm)))
    psi_p = np.exp(-2*(r1_p_norm + r2_p_norm) + r12_p_norm/(2*(1 + alpha * r12_p_norm)))

    return np.square(psi_p / psi)


def vmc(num_walkers, num_mc_steps, num_therm_steps, alpha, new_walker_std=1.0, save_walker_history=False):

    if num_mc_steps <= num_therm_steps:
        print("ERROR: Number of MC steps must be greater than number of thermalizing steps")
        return 0, 0, 0

    # Initialize walkers and energy
    walkers = np.random.rand(num_walkers, 3, 2)
    energy_estimator = np.zeros([num_mc_steps-num_therm_steps, num_walkers])
    num_accept_jumps = 0
    if save_walker_history:
        walker_history_xy = np.zeros([num_walkers, 2, 2, num_mc_steps])

    for step_mc in range(num_mc_steps):

        progress = 100*step_mc/num_mc_steps
        if progress % 5 == 0:
            print("Progress: " + str(progress) + "%")

        # Randomly create new walkers
        new_walkers = np.random.normal(walkers, scale=new_walker_std)

        # Test new walkers and update
        update_conditions = np.random.rand(num_walkers) < trans_prob(walkers, new_walkers, alpha)
        walkers[update_conditions, :, :] = new_walkers[update_conditions, :, :]
        num_accept_jumps += np.sum(update_conditions)

        # Evaluate local energy at walkers positions
        if step_mc >= num_therm_steps:
            energy_estimator[step_mc - num_therm_steps, :] = energy_local(walkers, alpha)

        if save_walker_history:
            walker_history_xy[:, :, :, step_mc] = walkers[:, 0:2, :]

    accept_ratio = num_accept_jumps / (num_walkers*num_mc_steps)
    if save_walker_history:
        return energy_estimator, accept_ratio, walker_history_xy
    else:
        return energy_estimator, accept_ratio, 0


if __name__ == "__main__":
    num_walk = 100
    num_steps = 20000
    therm_steps = 5000
    alphas = np.linspace(0.05, 0.25, num=2)
    energies = np.zeros(len(alphas))
    energies_std = np.zeros(len(alphas))
    accept_rats = np.zeros(len(alphas))
    walker_hist = np.zeros([num_walk, 2, 2, num_steps, len(alphas)])
    i = 0

    for a in alphas:
        print("alpha = " + str(a))

        energy_est, accept_rat, walker_hist[:, :, :, :, i] = vmc(num_walk, num_steps, therm_steps, alpha=a, save_walker_history=False)
        if np.isscalar(energy_est):
            break
        energies[i] = np.average(energy_est.flatten())
        energies_std[i] = np.std(energy_est.flatten())
        accept_rats[i] = accept_rat
        i += 1

    print(energies)
    print(accept_rats)
    print(walker_hist.shape)

    fig_energy = plt.figure(num=1)
    ax = fig_energy.add_subplot(111)
    ax.errorbar(alphas, energies, yerr=energies_std, ecolor="black", elinewidth=1.0, capsize=3,
                marker="*", color="red", linestyle="--", linewidth=1.0, label="VMC")
    ax.legend()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Energy estimation")

    animate = False
    if animate:
        fig_walkers = plt.figure(num=2)
        ax2 = fig_walkers.add_subplot(111)
        lim = 3
        for i in range(10000, 10500):
            ax2.clear()
            ax2.set_aspect("equal", "box")
            ax2.set_xlim(-lim, lim)
            ax2.set_ylim(-lim, lim)
            ax2.scatter(walker_hist[:, 0, 0, i, 1], walker_hist[:, 1, 0, i, 1], color="blue", s=3)
            ax2.scatter(walker_hist[:, 0, 1, i, 1], walker_hist[:, 1, 1, i, 1], color="red", s=3)
            plt.pause(0.001)
        plt.show()
