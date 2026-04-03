#!/bin/bash
#SBATCH --job-name="phenom-distribution-fit"
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH -t 05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --partition=
#SBATCH --account=

injection_attempt=""

for arg in "$@"; do
    case $arg in
        injection_attempt=*)
            injection_attempt="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: sbatch ../scripts/launch_inference_in_singularity_container.sh injection_attempt=<0|1|2|...>"
            exit 1
            ;;
    esac
done

if [ -z "$injection_attempt" ]; then
    echo "Missing required argument: injection_attempt"
    echo "Usage: sbatch ../scripts/launch_inference_in_singularity_container.sh injection_attempt=<0|1|2|...>"
    exit 1
fi

echo "Job is starting on $(hostname)"
echo "Job is starting from directory:"
echo "$PWD"
echo "Injection attempt: $injection_attempt"

project=phenom-distribution-fit
pathtoscratch=$HOME/scratch/$project/data/raw/
mkdir -p $pathtoscratch

pathtoscripts="$(realpath ../scripts)"

singularity exec --nv --bind $pathtoscratch,$PWD,$pathtoscripts ../singularity/phenom_distribution_fit_image.sif bash ../scripts/launch_inference.sh "$injection_attempt"
