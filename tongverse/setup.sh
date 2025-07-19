#!/bin/bash
# Create a conda environment
ISAAC_SIM_PATH=$HOME/.local/share/ov/pkg/isaac-sim-2023.1.1
ISAAC_PYTHON_VERSION="3.10.13"
echo "ISAAC_PYTHON_VERSION = ${ISAAC_PYTHON_VERSION}"

source $(conda info --base)/etc/profile.d/conda.sh
ENV_NAME="tongverse2023"
conda create -y -n "${ENV_NAME}" "python=${ISAAC_PYTHON_VERSION}"

conda activate ${ENV_NAME}
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
CONDA_ACTIVATE=${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh

# Set variable
echo "export TONGVERSE_RELEASE_URLBASE="http://10.2.31.187/tongverse-release"">>${CONDA_ACTIVATE}

echo "export ISAAC_SIM_PATH=$HOME/.local/share/ov/pkg/isaac_sim-2023.1.1">>${CONDA_ACTIVATE}
echo "source ${ISAAC_SIM_PATH}/setup_conda_env.sh" >> ${CONDA_ACTIVATE}
echo "source ${ISAAC_SIM_PATH}/setup_python_env.sh" >> ${CONDA_ACTIVATE}

mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
CONDA_DEACTIVATE=${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
echo "unset ISAAC_PATH" >> ${CONDA_DEACTIVATE}
echo "unset CARB_APP_PATH" >> ${CONDA_DEACTIVATE}
echo "unset EXP_PATH" >> ${CONDA_DEACTIVATE}
echo "unset PYTHONPATH" >> ${CONDA_DEACTIVATE}


pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# Install pre-commit if it's not alreay installed
pip install pre-commit -i https://pypi.tuna.tsinghua.edu.cn/simple


pip install -e .[dev] -i https://pypi.tuna.tsinghua.edu.cn/simple
# Check if pre-commit is installed correctly
if ! command -v pre-commit &> /dev/null
then
    echo "pre-commit could not be found"
    exit 1
fi


pre-commit install

# Install Orbit
# cd ..
# ORBIT_FOLDER="orbit"

# if [ -d "$ORBIT_FOLDER" ] && [ "$(ls -A "$ORBIT_FOLDER")" ]; then
#     echo " '$ORBIT_FOLDER' already exists and is not empty. Skip clone"
# else
#     git clone https://github.com/NVIDIA-Omniverse/orbit.git
#     # Sanity check
#     if [ $? -eq 0 ]; then
#         echo "Orbit clone successful"
#     else
#         echo "Orbit clone failed"
#         exit 1 # Exit with a non-zero status indicating failure
#     fi
# fi

# cd orbit
# # Clone the specific commit
# git reset --hard f6ec7e2185592c40ba7007c8a209cf3231d97f50
# export ISAACSIM_PATH=$HOME/.local/share/ov/pkg/isaac_sim-2023.1.1
# # Create a symbolic link
# ln -s ${ISAACSIM_PATH} _isaac_sim
# echo -e "alias orbit=$(pwd)/orbit.sh" >> $HOME/.bashrc
# source $HOME/.bashrc
# sudo apt install cmake build-essential
# ./orbit.sh -i

conda deactivate
echo -e "\nTongVerse successfully installed!"
