import numpy as np
import matplotlib.pyplot as plt


# Hamiltonian H = -(1/2)(d^2/dx^2) + (1/2)x^2
# Trial wave function psi_T = exp(-alpha*x^2)
# Local energy E_L = alpha + x^2(1/2 - 2*alpha^2)


def energy_local(x, alpha):
    return alpha + np.square(x) * (1 / 2 - 2 * np.square(alpha))


def trans_prob(x, x_p, alpha):
    return np.exp(-2 * alpha * (np.square(x_p) - np.square(x)))


def boot_strap(nr_data_points, nr_samples, quantity):
    # Calculates std's with bootstrapping
    random_choice = np.random.choice(quantity, (nr_data_points, nr_samples))
    sample = np.mean(random_choice, axis=0)
    return np.sqrt(nr_data_points - 1) * np.std(sample, ddof=1)


def vmc(num_walkers, num_mc_steps, num_therm_steps, alpha, new_walker_std=1.0):
    # Performs the MC calculations

    # Initialize walkers and energy
    walkers = np.random.rand(num_walkers) - 0.5
    energy_estimator = np.zeros(num_mc_steps - num_therm_steps)
    num_accept_jumps = 0

    for step_mc in range(num_mc_steps):

        progress = 100 * step_mc / num_mc_steps
        if progress % 5 == 0:
            print("Progress: " + str(progress) + "%")

        # Randomly create new walkers
        new_walkers = np.random.normal(walkers, scale=new_walker_std)

        # Test new walkers and update
        update_conditions = np.random.rand(num_walkers) < trans_prob(walkers, new_walkers, alpha)
        walkers[update_conditions] = new_walkers[update_conditions]
        num_accept_jumps += np.sum(update_conditions)

        # Evaluate local energy at walkers positions
        if step_mc >= num_therm_steps:
            energy_estimator[step_mc - num_therm_steps] = np.average(energy_local(walkers, alpha))

    accept_ratio = num_accept_jumps / (num_walkers * num_mc_steps)

    return energy_estimator, accept_ratio


if __name__ == "__main__":
    # Example on running calculations, sweeps over different values of alpha

    # Parameters
    num_walk = 100
    num_steps = 20000
    therm_steps = 5000
    alphas = np.linspace(0.1, 1, num=19)

    # Init
    energies = np.zeros(len(alphas))
    energies_error = np.zeros(len(alphas))
    accept_rats = np.zeros(len(alphas))
    i = 0

    for a in alphas:
        print("alpha = " + str(a))

        # Do Monte Carlo for current alpha
        energy_est, accept_rat = vmc(num_walk, num_steps, therm_steps, alpha=a)

        # Calc energy
        energies[i] = np.average(energy_est)

        # Bootstrapping error estimation
        nr_samples = 500
        nr_data_points = len(energy_est)
        energies_error[i] = boot_strap(nr_data_points=nr_data_points, nr_samples=nr_samples, quantity=energy_est)

        # Acceptance ratio
        accept_rats[i] = accept_rat

        i += 1

    print(energies)
    print(accept_rats)

    # Analytical solution
    alpha_analytical = np.linspace(alphas[0] - 0.01, alphas[-1] + 0.01, num=50)
    energies_analytical = 0.5 * alpha_analytical + np.divide(1, 8 * alpha_analytical)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alpha_analytical, energies_analytical, "black", linewidth=1.0, label="Analytical solution")
    ax.errorbar(alphas, energies, yerr=energies_error, ecolor="black", elinewidth=1.0, capsize=3,
                marker="*", color="red", linestyle="", linewidth=1.0, label="VMC")
    ax.legend(fontsize=14)
    ax.set_xticks(np.linspace(alphas[0], alphas[-1], num=10))
    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_ylabel("Energy estimation", fontsize=12)
    ax.set_title("Harmonic oscillator", fontsize=14)
    plt.show()
