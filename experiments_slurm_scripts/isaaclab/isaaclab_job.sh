#!/bin/bash

# Corrected SLURM script for IDS/TKS GPU Cluster
# Based on actual cluster configuration and IsaacLab requirements

# Function to create and submit a job for a single IsaacLab environment
submit_job() {
    local env_name=$1
    local job_name="isaaclab_${env_name}"
    
    # Create a temporary job script
    cat > "job_${env_name}.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err
#SBATCH --partition=short7d
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gpus=a100_80gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Set CUDA environment variables for CUDA 11.8
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH

# Initialize conda - adjust path if needed
source ~/.bashrc
source /mnt/home/daaboul/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Create output directory
mkdir -p outputs/\${SLURM_JOB_ID}

# Change to your FastTD3 directory - adjust path as needed
cd /mnt/home/\${USER}/workspaces/FastTD3/
export PYTHONPATH=/mnt/home/${USER}/workspaces/FastTD3/:$PYTHONPATH

# Run the training with IsaacLab-specific parameters
python -m fast_td3.train \\
    --env_name ${env_name} \\
    --exp_name ${env_name} \\
    --seed 0 \\
    --output_dir outputs/\${SLURM_JOB_ID} \\
    --headless \\
    --render_interval 0
EOF

    # Make the script executable
    chmod +x "job_${env_name}.slurm"
    
    # Submit the job
    sbatch "job_${env_name}.slurm"
    
    # Clean up the temporary script
    rm "job_${env_name}.slurm"
}

# Submit jobs for all IsaacLab environments
submit_job "Isaac-Lift-Cube-Franka-v0"
submit_job "Isaac-Open-Drawer-Franka-v0"
submit_job "Isaac-Velocity-Flat-H1-v0"
submit_job "Isaac-Velocity-Flat-G1-v0"