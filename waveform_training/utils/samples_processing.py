import jax
import jax.numpy as jnp
# ------------------------------------------------------------------
# Laptop settings:
# Force JAX to use the CPU, since a compatible GPU may be unavailable
# or may not have enough memory.
jax.config.update("jax_platforms", "cpu")
# ------------------------------------------------------------------
import pickle
import numpy as np
import os
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
plt.rcParams.update({"text.usetex": True})
import arviz as az
from utils.constants_jax import lambda_sigma_IMRPhenomD_33
from tqdm import tqdm


def get_original_chains_train(path_to_run):

    with open(path_to_run + '/parameter_transformation.pkl', 'rb') as f:
        linear_transformation_matrix = pickle.load(f)

    with open(path_to_run + '/train.pkl', 'rb') as file:
        out_train = pickle.load(file)

    chains_train = np.array(out_train['chains'])

    def inverse_custom_transform(transformed_params, linear_transformation_matrix):
        return jnp.dot(transformed_params, linear_transformation_matrix.T)
    
    original_chains_train = inverse_custom_transform(chains_train,linear_transformation_matrix)

    return original_chains_train

def get_original_chains_production(path_to_run):

    with open(path_to_run + '/parameter_transformation.pkl', 'rb') as f:
        linear_transformation_matrix = pickle.load(f)

    with open(path_to_run + '/summary.pkl', 'rb') as file:
        summary = pickle.load(file)

    chains, log_prob, local_accs, global_accs = summary.values()

    def inverse_custom_transform(transformed_params, linear_transformation_matrix):
        return jnp.dot(transformed_params, linear_transformation_matrix.T)
    
    original_chains_production = inverse_custom_transform(chains,linear_transformation_matrix)

    return original_chains_production

def save_mean_of_last_train_samples(path_to_run, outdir, n_last_samples=100):
    original_chains_train = get_original_chains_train(path_to_run)

    last_training_samples = original_chains_train[:, -n_last_samples:, :]
    flattened_last_training_samples = last_training_samples.reshape(
        -1, last_training_samples.shape[2]
    )

    mean_of_last_training_samples = jnp.mean(
        flattened_last_training_samples,
        axis=0,
    )

    run_label = os.path.basename(os.path.normpath(path_to_run))
    save_path = os.path.join(outdir, f"{run_label}_train_mean.pkl")

    with open(save_path, "wb") as file:
        pickle.dump(mean_of_last_training_samples, file)

    return mean_of_last_training_samples

def save_plot_final_r_hat(path_to_run, outdir):
    plt.figure(figsize=(15, 4))
    original_chains_production = get_original_chains_production(path_to_run)
    run_label = os.path.basename(os.path.normpath(path_to_run))

    original_chains_production_inference_data = az.convert_to_inference_data(np.array(original_chains_production))
    n_draws = original_chains_production_inference_data.posterior.dims['draw']
    n_dim = original_chains_production.shape[2]
    print(f" {n_draws} draws in {n_dim} dimensions")

    rhat = az.rhat(original_chains_production_inference_data)
    width = 0.35
    x = original_chains_production_inference_data.posterior.x_dim_0.data
    rhat_values = rhat.to_array().data.squeeze()
    plt.bar(x,  rhat_values, width)
    plt.axhline(1.0, color='k', linestyle='--')
    plt.xticks(x, ['x{:d}'.format(i) for i in x])
    plt.ylabel('$\hat{R}$')
    plt.title(f"Gelman-Rubin statistic computed from production chains of run {run_label}")
    plt.ylim(0, rhat_values.max() * 1.1)
    rhat_values_formatted = [f"{value:.5e}" for value in rhat_values]
    print(f"r hat values: {rhat_values_formatted}")

    for i, v in enumerate(rhat_values):
        plt.text(x[i] - width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
        
    save_path = os.path.join(outdir, f"{run_label}_final_r_hat.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return None

def save_plot_corner_with_reference_no_ticks(path_to_run, outdir, selected_indices):
    """
    Plot a corner plot for the selected parameter indices, with KDE contours,
    marginal distributions, and reference (IMRPhenomD) lines. Axis ticks are hidden.
    """
    run_label = os.path.basename(os.path.normpath(path_to_run))
    reference_array = np.asarray(lambda_sigma_IMRPhenomD_33)

    coords = {"x_dim_0": selected_indices}
    reference_values = {
        f"x {idx}": reference_array[idx]
        for idx in selected_indices
    }

    original_chains_production = get_original_chains_production(path_to_run)
    inference_data = az.convert_to_inference_data(
        np.asarray(original_chains_production)
    )

    axes = az.plot_pair(
        inference_data,
        kind="kde",
        coords=coords,
        kde_kwargs={
            "fill_last": False,
            "hdi_probs": [0.6, 0.9],
            "contourf_kwargs": {"cmap": plt.cm.viridis},
        },
        marginals=True,
        point_estimate="median",
        figsize=(15, 15),
        textsize=20,
        reference_values=reference_values,
        reference_values_kwargs={
            "color": "red",
            "linestyle": "--",
            "linewidth": 1.5,
        },
    )

    n = len(selected_indices)
    fig = axes[0, 0].figure

    for ax in axes.ravel():
        if ax is not None:
            ax.set_xticks([])
            ax.set_yticks([])

    for i, dim in enumerate(selected_indices):
        ax = axes[i, i]
        ax.axvline(
            x=reference_array[dim],
            color="red",
            linestyle="--",
            linewidth=2.0,
        )

    legend_handle = mlines.Line2D(
        [],
        [],
        color="red",
        linestyle="--",
        linewidth=6.0,
        label="LAL IMRPhenomD",
    )
    fig.legend(
        handles=[legend_handle],
        loc="upper right",
        bbox_to_anchor=(0.8, 0.8),
        prop={"size": 30},
    )

    fig.suptitle(
        f"Corner plot of production chains of run {run_label}",
        fontsize=30,y=0.92
    )

    save_path = os.path.join(
        outdir,
        f"{run_label}_{selected_indices}_corner.png",
    )
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return None

def save_plot_log_likelihood_along_train_chains(path_to_run, outdir, log_likelihood_function, data):
    """
    Evaluate and plot log likelihood values along selected thinned training chains.
    """
    run_label = os.path.basename(os.path.normpath(path_to_run))
    original_chains_train = get_original_chains_train(path_to_run)

    length_of_each_chain_after_thinning = 300
    selected_chains_indices = [0,92]
    
    chain_length = original_chains_train.shape[1]
    thinning_step = max(chain_length // length_of_each_chain_after_thinning, 1)
    thinned_chain = original_chains_train[:, ::thinning_step, :]
    thinned_chain = thinned_chain[:, :length_of_each_chain_after_thinning, :]
    
    log_likelihood_along_chain = np.zeros((len(selected_chains_indices), thinned_chain.shape[1]))

    for k, chain in enumerate(selected_chains_indices):
        for i, sample in enumerate(tqdm(thinned_chain[chain, :, :], desc=f"Chain {chain}")):
            log_likelihood_along_chain[k, i] = log_likelihood_function(sample, data)

        log_likelihood_along_chain[k, :] -= log_likelihood_along_chain[k, 0]
    
    plt.figure(figsize=(6, 4))
    for k in range(len(selected_chains_indices)):
        plt.plot(log_likelihood_along_chain[k, :], label=f'Chain {selected_chains_indices[k]}')

    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Log Likelihood", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gcf().set_size_inches(6, 4)

    save_path = os.path.join(outdir, run_label + '_loglikelihood_along_train_chain.png')
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    return None