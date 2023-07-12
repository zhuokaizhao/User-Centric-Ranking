# check if NVidia GPU is available
# if [[ $(lshw -C display | grep vendor) =~ "NVIDIA Corporation" ]]; then
echo "Installing PyTorch with CUDA"
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
# else
#     echo "Installing CPU-only PyTorch"
#     conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
# fi

# other dependencies
pip install -U deepctr-torch
conda install -c conda-forge pandas -y