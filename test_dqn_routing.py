import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
from collections import deque, namedtuple

# First, import the necessary classes from your main script
from packet_routing import ServerConfig, Packet, PacketRoutingEnv, DQN, DQNAgent

def setup_test_dirs():
    """Create directories for test results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'test_results_{timestamp}'
    dirs = {
        'base': base_dir,
        'plots': os.path.join(base_dir, 'plots'),
        'metrics': os.path.join(base_dir, 'metrics')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def load_trained_model(model_path, state_size, action_size, device):
    """Load the trained model"""
    model = DQN(state_size, action_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def test_model(model_path, test_data, server_configs, output_dirs):
    """Test the trained model on new data"""
    # Create test environment
    test_env = PacketRoutingEnv(test_data, server_configs)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create and load model
    state_size = test_env.state_size
    action_size = len(test_env.action_space)
    q_network = load_trained_model(model_path, state_size, action_size, device)
    
    # Initialize metrics
    test_metrics = {
        'step_balance_scores': [],
        'step_server_loads': [],
        'final_loads': [],
        'processed_packets': 0,
        'rejected_packets': 0,
        'server_assignments': [0] * len(server_configs),
        'server_utilization': []
    }
    
    # Run test episode
    state = test_env.reset()
    total_reward = 0
    step = 0
    
    print("\nStarting Test Evaluation...")
    print(f"Total packets to process: {len(test_env.data)}")
    
    while True:
        # Get action from trained model
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = q_network(state_tensor)
            action = torch.argmax(action_values).item()
        
        # Take action
        next_state, reward, done, info = test_env.step(action)
        total_reward += reward
        
        # Track metrics
        if info['processed']:
            test_metrics['processed_packets'] += 1
            test_metrics['server_assignments'][action] += 1
        else:
            test_metrics['rejected_packets'] += 1
        
        # Track server loads and balance
        server_loads = []
        for server in test_env.servers:
            cpu_util = server.used_cpu / server.max_cpu
            mem_util = server.used_memory / server.max_memory
            bw_util = server.used_bandwidth / server.max_bandwidth
            avg_load = (cpu_util * 0.4 + mem_util * 0.3 + bw_util * 0.3) * 100
            server_loads.append(avg_load)
        
        test_metrics['step_server_loads'].append(server_loads)
        
        # Calculate balance score
        if server_loads:
            avg_load = np.mean(server_loads)
            if avg_load > 0:
                std_dev = np.std(server_loads)
                balance_score = 1 - (std_dev / (avg_load + 1e-6))
            else:
                balance_score = 1.0
            test_metrics['step_balance_scores'].append(balance_score)
        
        # Track current server utilization
        test_metrics['server_utilization'].append([
            server.get_usage_percentages() for server in test_env.servers
        ])
        
        step += 1
        if step % 100 == 0:
            print(f"Processed {step} packets...")
        
        if done:
            break
        
        state = next_state
    
    # Store final loads
    test_metrics['final_loads'] = server_loads
    
    # Calculate final metrics
    total_packets = test_metrics['processed_packets'] + test_metrics['rejected_packets']
    acceptance_rate = test_metrics['processed_packets'] / total_packets if total_packets > 0 else 0
    
    # Print results
    print("\nTest Results:")
    print(f"Total Packets Processed: {test_metrics['processed_packets']}")
    print(f"Total Packets Rejected: {test_metrics['rejected_packets']}")
    print(f"Acceptance Rate: {acceptance_rate:.2%}")
    print(f"Total Reward: {total_reward:.2f}")
    print("\nServer Assignment Distribution:")
    for i, assignments in enumerate(test_metrics['server_assignments']):
        print(f"Server {i+1}: {assignments} packets ({assignments/total_packets:.2%})")
    
    # Save metrics and create plots
    save_test_results(test_metrics, output_dirs)
    create_test_plots(test_metrics, output_dirs)
    
    return test_metrics

def create_test_plots(metrics, output_dirs):
    """Create visualization of test results"""
    # Create first figure with time series plots
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Load Balance Score Over Steps
    plt.subplot(2, 1, 1)
    steps = range(len(metrics['step_balance_scores']))
    plt.plot(steps, metrics['step_balance_scores'], 'g-', label='Balance Score')
    plt.title('Load Balance Score Over Steps')
    plt.xlabel('Step (Packet Assignment)')
    plt.ylabel('Balance Score')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Server Utilization Over Steps
    plt.subplot(2, 1, 2)
    steps = range(len(metrics['step_server_loads']))
    for i in range(len(metrics['step_server_loads'][0])):
        server_loads = [step[i] for step in metrics['step_server_loads']]
        plt.plot(steps, server_loads, label=f'Server {i+1}')
    plt.title('Server Utilization Over Steps')
    plt.xlabel('Step (Packet Assignment)')
    plt.ylabel('Utilization (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['plots'], 'test_metrics.png'))
    plt.close()
    
    # Create second figure for load vs capacity bar graph
    plt.figure(figsize=(10, 6))
    server_indices = np.arange(len(metrics['final_loads']))
    bar_width = 0.35
    
    # Plot capacity bars (blue)
    capacities = [100] * len(server_indices)  # Maximum capacity normalized to 100%
    plt.bar(server_indices - bar_width/2, capacities, bar_width, 
            label='Server Capacity', color='skyblue')
    
    # Plot current load bars (red)
    plt.bar(server_indices + bar_width/2, metrics['final_loads'], bar_width,
            label='Current Load', color='salmon')
    
    plt.title('Current Load vs Server Capacity')
    plt.xlabel('Server')
    plt.ylabel('Utilization (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(server_indices, [f'Server {i+1}' for i in server_indices])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['plots'], 'load_vs_capacity.png'))
    plt.close()

def save_test_results(metrics, output_dirs):
    """Save test results to file"""
    test_summary = {
        'processed_packets': metrics['processed_packets'],
        'rejected_packets': metrics['rejected_packets'],
        'acceptance_rate': metrics['processed_packets'] / 
                         (metrics['processed_packets'] + metrics['rejected_packets']),
        'server_assignments': {
            f'server_{i+1}': count for i, count in enumerate(metrics['server_assignments'])
        },
        'final_server_loads': {
            f'server_{i+1}': load for i, load in enumerate(metrics['final_loads'])
        },
        'average_balance_score': np.mean(metrics['step_balance_scores']),
        'server_utilization_stats': {
            f'server_{i+1}': {
                'mean_load': np.mean([loads[i] for loads in metrics['step_server_loads']]),
                'max_load': np.max([loads[i] for loads in metrics['step_server_loads']]),
                'min_load': np.min([loads[i] for loads in metrics['step_server_loads']])
            } for i in range(len(metrics['step_server_loads'][0]))
        }
    }
    
    with open(os.path.join(output_dirs['metrics'], 'test_results.json'), 'w') as f:
        json.dump(test_summary, f, indent=4)

def main():
    # Setup directories for test results
    output_dirs = setup_test_dirs()
    
    # Load test data
    test_data = pd.read_csv('network_traffic_subset.csv').sample(n=25, random_state=42)
    
    # Define server configurations (same as training)
    server_configs = [
        {'cpu': 2.0, 'memory': 4, 'bandwidth': 1000},  # 2GHz CPU, 4GB RAM, 1Gbps
        {'cpu': 3.0, 'memory': 8, 'bandwidth': 2000},  # 3GHz CPU, 8GB RAM, 2Gbps
        {'cpu': 1.5, 'memory': 2, 'bandwidth': 500}    # 1.5GHz CPU, 2GB RAM, 500Mbps
    ]
    
    # Path to your trained model
    model_path = 'final_model.pth'
    
    # Run test
    test_metrics = test_model(model_path, test_data, server_configs, output_dirs)
    print(f"\nTest results saved in: {output_dirs['base']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
