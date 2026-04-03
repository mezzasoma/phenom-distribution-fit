#! /bin/bash

. ~/.bashrc
conda env create -f ./conda/environment_gwsurrogate.yml
conda activate gw-surrogate

read -rp "Create an IPython kernel named 'gw-surrogate'? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
  python -m ipykernel install --user --name=gw-surrogate
fi

echo
echo "If you need to remove the conda environment and the IPython kernel later, run:"
echo "  conda remove -n gw-surrogate --all"
echo "  jupyter kernelspec uninstall gw-surrogate"