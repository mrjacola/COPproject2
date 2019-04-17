import numpy as np
import matplotlib.pyplot as plt


# Hamiltonian H = -(1/2)(d^2/dx^2) + (1/2)x^2
# Trial wavefunction psi_T = exp(-alpha*x^2)
# Local energy E_L = alpha + x^2(1/2 - 2*alpha^2)


def energy_local(x, alpha):
    return alpha + np.square(x)*(1/2 - 2*np.square(alpha))


def trans_prob(x, x_p, alpha):
    return np.exp(-2*alpha*(np.square(x_p) - np.square(x)))


def vmc(num_walkers, num_MC_steps, num_therm_steps, alpha, new_walker_std=1.0):

    if num_MC_steps <= num_therm_steps:
        print("ERROR: Number of steps must be greater than number of thermalizing steps")
        return 0, 0

    # Initialize walkers and energy
    walkers = np.random.rand(num_walkers) - 0.5
    energy_estimator = np.zeros([num_MC_steps-num_therm_steps, num_walkers])
    num_accept_jumps = 0

    for step_MC in range(num_MC_steps):

        progress = 100*step_MC/num_MC_steps
        if progress % 5 == 0:
            print("Progress: " + str(progress) + "%")

        # Randomly create new walkers
        new_walkers = np.random.normal(walkers, scale=new_walker_std)

        # Test new walkers and update
        update_conditions = np.random.rand(num_walkers) < trans_prob(walkers, new_walkers, alpha)
        walkers[update_conditions] = new_walkers[update_conditions]
        num_accept_jumps += np.sum(update_conditions)

        # Evaluate local energy at walkers positions
        if step_MC >= num_therm_steps:
            energy_estimator[step_MC - num_therm_steps, :] = energy_local(walkers, alpha)

    accept_ratio = num_accept_jumps / (num_walkers*num_MC_steps)

    return energy_estimator, accept_ratio


if __name__ == "__main__":
    num_walk = 100
    num_steps = 20000
    therm_steps = 5000
    alphas = np.linspace(0.1, 1.5, num=30)
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