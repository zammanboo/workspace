#!/bin/bash

# NCCL Test Script for 32 nodes with 8 B200 GPUs each
# Total GPUs: 256 (32 nodes × 8 GPUs)

set -e

# Configuration
NODES=32
GPUS_PER_NODE=8
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
MASTER_PORT=29500
NCCL_TEST_PATH="/opt/nccl-tests/build"  # Adjust path as needed

# Test parameters
TEST_SIZE="1G"  # Test data size
WARMUP_ITERS=5
ITERS=20

echo "========================================="
echo "NCCL Test Configuration"
echo "========================================="
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Test size: $TEST_SIZE"
echo "Iterations: $ITERS"
echo "Warmup iterations: $WARMUP_ITERS"
echo "========================================="

# Function to get master node IP
get_master_ip() {
    if [ -n "$SLURM_JOB_ID" ]; then
        # SLURM environment
        echo $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
    elif [ -n "$PBS_NODEFILE" ]; then
        # PBS environment
        echo $(head -n1 $PBS_NODEFILE)
    else
        # Default to localhost for testing
        echo "localhost"
    fi
}

MASTER_ADDR=$(get_master_ip)
echo "Master address: $MASTER_ADDR"

# NCCL Environment variables for B200 optimization
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Starting NCCL tests..."

# Function to run NCCL test
run_nccl_test() {
    local test_name=$1
    local test_binary=$2
    
    echo "----------------------------------------"
    echo "Running $test_name"
    echo "----------------------------------------"
    
    if [ -n "$SLURM_JOB_ID" ]; then
        # SLURM job
        srun --ntasks=$NODES \
             --ntasks-per-node=1 \
             --gpus-per-task=$GPUS_PER_NODE \
             $NCCL_TEST_PATH/$test_binary \
             -b $TEST_SIZE -e $TEST_SIZE -f 2 -g $GPUS_PER_NODE \
             -w $WARMUP_ITERS -n $ITERS
    elif [ -n "$PBS_NODEFILE" ]; then
        # PBS job
        mpirun -np $NODES \
               -hostfile $PBS_NODEFILE \
               -map-by node:PE=$GPUS_PER_NODE \
               $NCCL_TEST_PATH/$test_binary \
               -b $TEST_SIZE -e $TEST_SIZE -f 2 -g $GPUS_PER_NODE \
               -w $WARMUP_ITERS -n $ITERS
    else
        # Manual execution (for testing)
        echo "Manual execution mode - adjust for your cluster setup"
        mpirun -np $NODES \
               -H localhost:$NODES \
               $NCCL_TEST_PATH/$test_binary \
               -b $TEST_SIZE -e $TEST_SIZE -f 2 -g $GPUS_PER_NODE \
               -w $WARMUP_ITERS -n $ITERS
    fi
}

# Run different NCCL tests
echo "Testing AllReduce..."
run_nccl_test "AllReduce" "all_reduce_perf"

echo "Testing Broadcast..."
run_nccl_test "Broadcast" "broadcast_perf"

echo "Testing AllGather..."
run_nccl_test "AllGather" "all_gather_perf"

echo "Testing ReduceScatter..."
run_nccl_test "ReduceScatter" "reduce_scatter_perf"

echo "Testing SendRecv..."
run_nccl_test "SendRecv" "sendrecv_perf"

echo "========================================="
echo "NCCL tests completed successfully!"
echo "========================================="

# Optional: Generate summary report
echo "Test Summary:" > nccl_test_results.txt
echo "Nodes: $NODES" >> nccl_test_results.txt
echo "GPUs per node: $GPUS_PER_NODE" >> nccl_test_results.txt
echo "Total GPUs: $TOTAL_GPUS" >> nccl_test_results.txt
echo "Test completed at: $(date)" >> nccl_test_results.txt