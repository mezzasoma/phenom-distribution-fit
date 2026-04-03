# phenom-distribution-fit
This repository contains instructions and scripts to reproduce the waveform calibration described in [_Uncertainty-aware waveform modeling for high signal-to-noise ratio gravitational-wave inference_](https://arxiv.org/abs/2503.23304). It uses the Python libraries [`ripplegw`](https://github.com/tedwards2412/ripple) (waveform generation) and [`flowMC`](https://github.com/kazewong/flowMC) (Bayesian inference assisted by normalizing flow). Both are accelerated on GPU with [`jax`](https://github.com/jax-ml/jax).

<video controls>
  <source src="sampling_animation.mp4" type="video/mp4">
</video>

## Instructions
1. From within the repository `phenom-distribution-fit`, create the conda environment `gw-surrogate` to generate the numerical relativity surrogate waveforms `NRHybSur3dq8`
    ```
    bash conda/create_gwsurrogate.sh
    conda activate gw-surrogate
    cd waveform_training
    python NRHybSur3dq8_training_set_frequency_domain.py
    cd ..
    conda deactivate
    ```
    The training set is made up of 94 `NRHybSur3dq8` waveforms.

2. Create the `phenom-distribution-fit` conda environment, used to post-process the inferred samples and to run convergence diagnostics 
    ```
    bash conda/create_phenom_distribution_fit.sh
    ```

3. Inference is intended to be executed on GPU. The safest way to ensure compatibility between `jax`, `ripplegw`, `flowMC`, and the NVIDIA libraries is to use a Singularity container.
Build the container with
    ```
    sudo singularity build singularity/phenom_distribution_fit_image.sif singularity/phenom_distribution_fit_image.def
    ```
    This requires root privileges and it can temporarily use up to ~17GB of disk space under root. This space is realease once the container is created. The final .sif image is ~4.2GB.

4. Remove temporary installation files
    ```
    bash conda/remove_temporary_files.sh
    ```

5. Fill in the user-dependent Slurm job settings
    ```
    #SBATCH --mail-user=
    #SBATCH --partition=
    #SBATCH --account=
    ```
    in the SBATCH script example
`scripts/launch_inference_in_singularity_container_example.sh`
and then rename the file to `scripts/launch_inference_in_singularity_container.sh`. This file will be ignored by `git`.

6. For the waveform training set considered here (~40MB), it is preferable to split the inference into multiple segments to avoid exhausting the GPU memory. Each segment is labeled by an integer `injection_attempt`, starting from 0. At the start of each segment, a linear parameter transformation is computed to decorrelate the parameters and improve sampling convergence.

    - 6.1 Compue the parameter transformation for `injection_attempt=0`
        ```
        cd waveform_training
        conda activate phenom-distribution-fit
        python parameter_transformation.py 0
        ```
    - 6.2 Submit the first inference job
        ```
        sbatch ../scripts/launch_inference_in_singularity_container.sh injection_attempt=0
        ```
    - 6.3 The inference output is saved under `$HOME/scratch/phenom-distribution-fit/data/raw/flowmc-YYYYMMDDHHMMSS`. Let `/path/to/run/flowmc-YYYYMMDDHHMMSS` denote the full path to the run directory, or more generally any path where the run is stored. Inspect this run with
        ```
        python inspect_inference.py path/to/run/flowmc-YYYYMMDDHHMMSS
        ```
        This saves the next injection point, computed from the mean of the last 100 samples produced during the learning phase of `flowMC`. It also saves plots to assess the
        convergence of the sampler: the log likelihood along the chains during the learning phase; the Gelman-Rubin statistic and a corner plot of the chains during the production phase.
    - 6.4 Compute the parameter transformation for `injection_attempt=1` by passing the previous run label `flowmc-YYYYMMDDHHMMSS`
        ```
        python parameter_transformation.py 1 flowmc-YYYYMMDDHHMMSS
        ```
    - 6.5 Launch the next inference segment, starting the sampler from the new injection point
        ```
        sbatch ../scripts/launch_inference_in_singularity_container.sh injection_attempt=1
        ```
        and then inspect the new run, here denoted as `flowmc-ZZZZFFWWQQJJLL`:
        ```
        python inspect_inference.py /path/to/run/flowmc-ZZZZFFWWQQJJLL
        ```
    Repeat steps 6.4 and 6.5 for `injection_attempt=2,3,4,...` until convergence is achieved (saturation of the log likelihood values, Gelman-Rubin statistic less than 1.05, good chain mixing). In practice, run through at least `injection_attempt=2`.

## Software and hardware requirements

This project was developed and tested with:

- [SingularityCE 3.11.0](https://docs.sylabs.io/guides/3.11/user-guide/)
- [conda 23.3.1](https://docs.conda.io/projects/conda/en/23.3.x/)
- [Slurm 25.05.5](https://slurm.schedmd.com/quickstart.html)
- NVIDIA GPU: [A100 80GB](https://www.nvidia.com/en-us/data-center/a100/) or [H200 141GB](https://www.nvidia.com/en-us/data-center/h200/)


