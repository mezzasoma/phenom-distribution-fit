#! /bin/bash

. ~/.bashrc
git clone https://github.com/tedwards2412/ripple.git ../ripple-phenom-distribution-fit-temp
git -C ../ripple-phenom-distribution-fit-temp checkout 320fcaa3fca680e12a1df6ec6213f54a9a19db53
cp ./conda/scripts_to_replace/ripple/IMRPhenomD.py ../ripple-phenom-distribution-fit-temp/src/ripple/waveforms/IMRPhenomD.py
echo
echo "src/ripple/waveforms/IMRPhenomD.py has been modified to sample 33 inspiral fitting coefficients"
cp ./conda/scripts_to_replace/ripple/setup.cfg ../ripple-phenom-distribution-fit-temp/setup.cfg
echo
echo "setup.cfg has been modified to avoid reinstalling jax and jaxlib"
echo

conda env create -f conda/environment_phenom_distribution_fit.yml
conda activate phenom-distribution-fit
pip3 install --force-reinstall --ignore-installed --no-cache-dir ../ripple-phenom-distribution-fit-temp
conda install -y ipykernel
pip install arviz==0.17.1

read -rp "Create an IPython kernel named 'phenom-distribution-fit'? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
  python -m ipykernel install --user --name=phenom-distribution-fit
fi

conda deactivate

echo
echo "If you need to remove the conda environment and the IPython kernel later, run:"
echo "  conda remove -n phenom-distribution-fit --all"
echo "  jupyter kernelspec uninstall phenom-distribution-fit"