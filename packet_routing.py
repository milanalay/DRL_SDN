import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

class ServerConfig:
    def __init__(self, cpu, memory, bandwidth):
        # Store resource capacities in base units
        self.max_cpu = cpu * 1e9        # Convert GHz to Hz
        self.max_memory = memory * 1e9   # Convert GB to bytes
        self.max_bandwidth = bandwidth * 1e6  # Convert Mbps to bps
        
        # Initialize current resource usage
        self.reset()
        
        # Resource thresholds for load balancing
        self.cpu_threshold = 0.75  # 75% CPU utilization threshold
        self.memory_threshold = 0.75  # 75% memory utilization threshold
        self.bandwidth_threshold = 0.70  # 70% bandwidth utilization threshold
    
    def reset(self):
        """Reset server state"""
        self.used_cpu = 0
        self.used_memory = 0
        self.used_bandwidth = 0
        self.processed_packets = 0
        
    def can_process(self, packet):
        """Check if server has enough resources to process the packet"""
        # Calculate remaining resources
        remaining_cpu = self.max_cpu - self.used_cpu
        remaining_memory = self.max_memory - self.used_memory
        remaining_bandwidth = self.max_bandwidth - self.used_bandwidth
        
        # Check if any resource would exceed threshold
        cpu_ok = (self.used_cpu + packet.cpu_requirement) / self.max_cpu <= self.cpu_threshold
        memory_ok = (self.used_memory + packet.memory_requirement) / self.max_memory <= self.memory_threshold
        bandwidth_ok = (self.used_bandwidth + packet.bandwidth_requirement) / self.max_bandwidth <= self.bandwidth_threshold
        
        # Resource availability check
        resources_available = (
            packet.cpu_requirement <= remaining_cpu and
            packet.memory_requirement <= remaining_memory and
            packet.bandwidth_requirement <= remaining_bandwidth
        )
        
        # Consider accepting if resources are available and won't exceed thresholds
        return resources_available and cpu_ok and memory_ok and bandwidth_ok

    def get_balanced_load_factor(self):
        """Calculate a balanced load factor considering all resources"""
        cpu_util = self.used_cpu / self.max_cpu
        memory_util = self.used_memory / self.max_memory
        bandwidth_util = self.used_bandwidth / self.max_bandwidth
        
        # Weight the utilizations
        weights = [0.4, 0.3, 0.3]  # More weight on CPU
        weighted_util = (
            weights[0] * cpu_util +
            weights[1] * memory_util +
            weights[2] * bandwidth_util
        )
        
        # Calculate imbalance penalty
        utilizations = [cpu_util, memory_util, bandwidth_util]
        std_dev = np.std(utilizations)
        imbalance_penalty = std_dev * 0.5
        
        # Return balanced score (lower is better)
        return weighted_util + imbalance_penalty
        
    def add_packet(self, packet):
        """Attempt to add packet to server"""
        if self.can_process(packet):
            self.used_cpu += packet.cpu_requirement
            self.used_memory += packet.memory_requirement
            self.used_bandwidth += packet.bandwidth_requirement
            self.processed_packets += 1
            return True
        return False
    
    def get_usage_percentages(self):
        """Calculate current resource usage as percentages"""
        return {
            'cpu': (self.used_cpu / self.max_cpu) * 100 if self.max_cpu > 0 else 100,
            'memory': (self.used_memory / self.max_memory) * 100 if self.max_memory > 0 else 100,
            'bandwidth': (self.used_bandwidth / self.max_bandwidth) * 100 if self.max_bandwidth > 0 else 100
        }
    
    def get_available_capacity(self):
        """Calculate available capacity as the minimum percentage available across all resources"""
        usage = self.get_usage_percentages()
        return 100 - max(usage.values())

    @property
    def cpu_usage_pct(self):
        return (self.used_cpu / self.max_cpu) * 100 if self.max_cpu > 0 else 100

    @property
    def memory_usage_pct(self):
        return (self.used_memory / self.max_memory) * 100 if self.max_memory > 0 else 100

    @property
    def bandwidth_usage_pct(self):
        return (self.used_bandwidth / self.max_bandwidth) * 100 if self.max_bandwidth > 0 else 100

    @property
    def available_capacity(self):
        return self.get_available_capacity()

class Packet:
    def __init__(self, row):
        # Extract features from the flow dataset
        self.pkt_count = row['pktTotalCount']
        self.octet_count = row['octetTotalCount']
        self.flow_duration = row['flowDuration']
        self.protocol = row['proto']
        
        # Calculate actual resource requirements
        self.calculate_resource_requirements()

    def calculate_resource_requirements(self):
        """Calculate actual resource requirements based on packet characteristics"""
        # Normalize flow duration (minimum 1ms)
        min_duration = max(self.flow_duration, 1000)  # microseconds to milliseconds
        
        # CPU requirement (Hz)
        # Increase base CPU requirements
        base_cpu_per_packet = 1e6 if self.protocol == 6 else 0.5e6  # 1MHz for TCP, 500KHz for UDP
        processing_factor = math.log1p(self.octet_count) / math.log1p(1500)  # Scale with packet size
        self.cpu_requirement = base_cpu_per_packet * self.pkt_count * processing_factor
        
        # Memory requirement (bytes)
        # Increase memory requirements
        packet_overhead = 4096  # 4KB per packet overhead
        buffer_multiplier = 2.5  # Increased buffer size
        self.memory_requirement = (
            (packet_overhead * self.pkt_count) +  # Control structures
            (self.octet_count * buffer_multiplier)  # Buffer for packet data
        ) * processing_factor
        
        # Bandwidth requirement (bps)
        # Reduce bandwidth requirements
        packet_overhead_bits = self.pkt_count * 320  # 40 bytes header per packet
        total_bits = (self.octet_count * 8) + packet_overhead_bits
        # Add rate limiting factor
        rate_limit_factor = 0.7  # Reduce bandwidth usage to 70%
        self.bandwidth_requirement = (total_bits * 1e6 / min_duration) * rate_limit_factor
        
        # Scale requirements to stay within reasonable limits
        max_cpu_per_server = 3e9  # 3 GHz
        max_memory_per_server = 8e9  # 8 GB
        max_bandwidth_per_server = 2e9  # 2 Gbps
        
        # Increase resource utilization caps
        self.cpu_requirement = min(self.cpu_requirement, max_cpu_per_server * 0.3)  # 30% max CPU
        self.memory_requirement = min(self.memory_requirement, max_memory_per_server * 0.25)  # 25% max memory
        self.bandwidth_requirement = min(self.bandwidth_requirement, max_bandwidth_per_server * 0.2)  # 20% max bandwidth
        
        # Calculate percentage requirements for RL state
        self.cpu_usage_pct = (self.cpu_requirement / max_cpu_per_server) * 100
        self.memory_usage_pct = (self.memory_requirement / max_memory_per_server) * 100
        self.bandwidth_usage_pct = (self.bandwidth_requirement / max_bandwidth_per_server) * 100
        
        # Adjust total capacity requirement calculation
        weights = [0.4, 0.3, 0.3]  # Give more weight to CPU
        self.capacity_requirement = max(
            0.5,  # Increased minimum requirement
            min(
                25.0,  # Increased maximum cap
                weights[0] * self.cpu_usage_pct +
                weights[1] * self.memory_usage_pct +
                weights[2] * self.bandwidth_usage_pct
            )
        )

class PacketRoutingEnv:
    def __init__(self, data, server_configs):
        self.full_dataset = [Packet(row) for _, row in data.iterrows()]
        self.servers = [ServerConfig(**config) for config in server_configs]
        self.action_space = list(range(len(self.servers)))
        self.state_size = self._calculate_state_size()
        self.data = []
        self.current_index = 0
        self.total_processed = 0
        self.total_rejected = 0

    def _calculate_state_size(self):
        # State includes server states, packet states, and global states
        return len(self.servers) * 4 + 4 + 4
    
    def reset(self):
        """Reset the environment to initial state"""
        # Sample a subset of packets for this episode
        sample_size = min(len(self.full_dataset), 500)
        self.data = random.sample(self.full_dataset, sample_size)
        self.current_index = 0
        self.total_processed = 0
        self.total_rejected = 0
        
        # Reset all servers
        for server in self.servers:
            server.reset()
            
        return self.get_state()

    def step(self, action):
        if self.current_index >= len(self.data):
            return self.get_state(), 0, True, {'termination_reason': 'completed'}

        packet = self.data[self.current_index]
        server = self.servers[action]
        
        processed = server.add_packet(packet)
        if processed:
            self.total_processed += 1
        else:
            self.total_rejected += 1
            
        reward = self.get_reward(action, processed)
        self.current_index += 1
        done = (self.current_index >= len(self.data))
        
        info = {
            'processed': processed,
            'current_packet': self.current_index,
            'total_packets': len(self.data),
            'processed_packets': self.total_processed,
            'rejected_packets': self.total_rejected,
            'server_loads': [100 - s.available_capacity for s in self.servers],
            'server_overhead': self.calculate_server_overhead(action),
            'global_overhead': self.calculate_global_overhead()
        }

        return self.get_state(), reward, done, info

    def calculate_server_overhead(self, server_idx):
        """Calculate overhead metric for a specific server using balanced load factor"""
        server = self.servers[server_idx]
        
        # Get individual resource utilizations
        cpu_util = server.used_cpu / server.max_cpu
        mem_util = server.used_memory / server.max_memory
        bw_util = server.used_bandwidth / server.max_bandwidth
        
        # Calculate average utilization
        avg_util = (cpu_util + mem_util + bw_util) / 3
        
        if avg_util == 0:  # Penalize unused servers
            return 3.0
            
        if avg_util >= 0.75:  # Penalize overloaded servers
            return 2.0
            
        if avg_util < 0.1:  # Penalize underutilized servers
            return 1.5
            
        # Calculate utilization imbalance
        utils = [cpu_util, mem_util, bw_util]
        std_dev = np.std(utils)
        balance_factor = std_dev / (avg_util + 1e-6)  # Avoid division by zero
        
        # Return balanced score (lower is better)
        return 1.0 + balance_factor

    def calculate_global_overhead(self):
        """Calculate overhead metric across all servers"""
        server_utils = []
        for server in self.servers:
            cpu_util = server.used_cpu / server.max_cpu
            mem_util = server.used_memory / server.max_memory
            bw_util = server.used_bandwidth / server.max_bandwidth
            avg_util = (cpu_util + mem_util + bw_util) / 3
            server_utils.append(avg_util)
        
        avg_util = np.mean(server_utils)
        if avg_util >= 0.75:
            return 2.0
        
        if avg_util < 0.1:
            return 1.5
            
        # Calculate distribution imbalance across servers
        std_dev = np.std(server_utils)
        return 1.0 + (std_dev * 2.0)  # Increased penalty for server imbalance

    def get_reward(self, action, processed):
        if not processed:
            if self._check_resource_exhaustion():
                return -0.5  # Reduced penalty for rejection when resources are exhausted
            return -1.0  # Standard rejection penalty
        
        # Base reward for successful processing
        base_reward = 5.0  # Increased base reward
        
        # Get server-specific metrics
        server = self.servers[action]
        cpu_util = server.used_cpu / server.max_cpu
        mem_util = server.used_memory / server.max_memory
        bw_util = server.used_bandwidth / server.max_bandwidth
        
        # Calculate balanced utilization bonus
        utils = [cpu_util, mem_util, bw_util]
        avg_util = np.mean(utils)
        std_dev = np.std(utils)
        
        # Reward for balanced resource usage
        balance_bonus = 2.0 * (1.0 - std_dev)
        
        # Reward for optimal utilization level (target: 40-60%)
        util_bonus = 2.0 * (1.0 - abs(0.5 - avg_util))
        
        # Get overhead metrics
        server_overhead = self.calculate_server_overhead(action)
        global_overhead = self.calculate_global_overhead()
        
        # Calculate final reward components
        server_reward = 3.0 * (3.0 - server_overhead)
        global_reward = 2.0 * (3.0 - global_overhead)
        
        return base_reward + balance_bonus + util_bonus + server_reward + global_reward

    def _check_resource_exhaustion(self):
        """Check if servers are truly near capacity"""
        for server in self.servers:
            # Check if any server has reasonable capacity left
            cpu_avail = 1 - (server.used_cpu / server.max_cpu)
            mem_avail = 1 - (server.used_memory / server.max_memory)
            bw_avail = 1 - (server.used_bandwidth / server.max_bandwidth)
            
            if all(avail > 0.25 for avail in [cpu_avail, mem_avail, bw_avail]):
                return False
        return True

    def get_state(self):
        if self.current_index >= len(self.data):
            return np.zeros(self.state_size)

        packet = self.data[self.current_index]
        
        # Include balanced load factors in server states
        server_states = []
        for server in self.servers:
            balanced_load = server.get_balanced_load_factor()
            server_states.extend([
                server.cpu_usage_pct / 100.0,
                server.memory_usage_pct / 100.0,
                server.bandwidth_usage_pct / 100.0,
                balanced_load  # Replace available_capacity with balanced_load
            ])

        packet_states = [
            packet.cpu_usage_pct / 100.0,
            packet.memory_usage_pct / 100.0,
            packet.bandwidth_usage_pct / 100.0,
            packet.capacity_requirement / 100.0
        ]

        # Use balanced load factors for global state
        load_factors = [server.get_balanced_load_factor() for server in self.servers]
        avg_load = np.mean(load_factors)
        load_std = np.std(load_factors)
        
        global_states = [
            avg_load,
            load_std,
            self.total_processed / max(1, self.current_index),
            self.calculate_global_overhead()
        ]

        return np.array(server_states + packet_states + global_states)

    

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            module.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.update_target_every = 50
        self.step_counter = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Smart exploration using state information
            # Extract server states from the full state vector
            num_servers = (len(state) - 8) // 4  # 4 values per server, 8 for packet and global states
            server_loads = []
            
            for i in range(num_servers):
                # Each server has [cpu, memory, bandwidth, balanced_load]
                start_idx = i * 4
                cpu_util = state[start_idx]
                mem_util = state[start_idx + 1]
                bw_util = state[start_idx + 2]
                avg_load = (cpu_util + mem_util + bw_util) / 3
                server_loads.append((i, avg_load))
            
            # Filter servers below 75% utilization
            valid_servers = [(idx, load) for idx, load in server_loads if load < 0.75]
            
            if valid_servers:
                if random.random() < 0.3:  # 30% chance to pick least loaded
                    least_loaded = min(valid_servers, key=lambda x: x[1])
                    return least_loaded[0]
                # Otherwise randomly choose from valid servers
                return random.choice([idx for idx, _ in valid_servers])
            
            # If no valid servers, choose randomly
            return random.randint(0, self.action_size - 1)
        
        # Use Q-values for exploitation
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.q_network(state)
            # Apply softmax to get probabilities
            probs = torch.softmax(action_values, dim=1)
            # Choose action with highest probability
            return torch.argmax(probs).item()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Get next Q values from target network
        with torch.no_grad():
            # Get best actions from current network (Double DQN)
            best_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Get Q values from target network for those actions
            next_q_values = self.target_network(next_states).gather(1, best_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss using Huber loss (more stable than MSE)
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

# Create output directories
def setup_output_dirs():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'training_results_{timestamp}'
    dirs = {
        'base': base_dir,
        'plots': os.path.join(base_dir, 'plots'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'models': os.path.join(base_dir, 'models')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

# Modify DQNAgent's train method to save metrics
def train_and_evaluate(train_data, test_data, server_configs, output_dirs, num_episodes):
    """Enhanced training and evaluation pipeline with comprehensive metrics saving"""
    # Create environments
    train_env = PacketRoutingEnv(train_data, server_configs)
    test_env = PacketRoutingEnv(test_data, server_configs)
    
    # Create agent
    state_size = train_env.state_size
    action_size = len(train_env.action_space)
    agent = DQNAgent(state_size, action_size)
    
    # Training metrics
    all_metrics = {
        'training_rewards': [],
        'average_rewards':[],
        'episode_lengths': [],
        'acceptance_rates': [],
        'server_loads': [],
        'episode_summaries': []
    }
    
    print("Starting Training...")
    for episode in range(num_episodes):
        state = train_env.reset()
        episode_metrics = {
            'steps': 0,
            'total_reward': 0,
            'processed_packets': 0,
            'rejected_packets': 0,
            'server_metrics': [{
                'cpu_usage': [],
                'memory_usage': [],
                'bandwidth_usage': [],
                'processed_packets': 0
            } for _ in server_configs]
        }
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = train_env.step(action)
            
            # Store transition
            agent.memory.add(state, action, reward, next_state, done)
            
            # Update episode metrics
            episode_metrics['steps'] += 1
            episode_metrics['total_reward'] += reward
            if info['processed']:
                episode_metrics['processed_packets'] += 1
                episode_metrics['server_metrics'][action]['processed_packets'] += 1
            else:
                episode_metrics['rejected_packets'] += 1
            
            # Update server metrics
            for i, server in enumerate(train_env.servers):
                episode_metrics['server_metrics'][i]['cpu_usage'].append(server.cpu_usage_pct)
                episode_metrics['server_metrics'][i]['memory_usage'].append(server.memory_usage_pct)
                episode_metrics['server_metrics'][i]['bandwidth_usage'].append(server.bandwidth_usage_pct)
            
            # Learn if enough samples
            if len(agent.memory) > agent.batch_size:
                experiences = agent.memory.sample(agent.batch_size)
                agent.learn(experiences)
            
            if done:
                break
            
            state = next_state
        
        # Update exploration rate
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Save episode metrics
        all_metrics['training_rewards'].append(episode_metrics['total_reward'])
        # Calculate and store moving average
        window_size = 20  # You can adjust this window size
        if len(all_metrics['training_rewards']) >= window_size:
            avg_reward = np.mean(all_metrics['training_rewards'][-window_size:])
        else:
            avg_reward = np.mean(all_metrics['training_rewards'])
        all_metrics['average_rewards'].append(avg_reward)

        all_metrics['episode_lengths'].append(episode_metrics['steps'])
        all_metrics['acceptance_rates'].append(
            episode_metrics['processed_packets'] / 
            (episode_metrics['processed_packets'] + episode_metrics['rejected_packets'])
        )
        
        # Save server loads at end of episode
        all_metrics['server_loads'].append([
            100 - server.available_capacity for server in train_env.servers
        ])
        
        # Save detailed episode summary
        all_metrics['episode_summaries'].append({
            'episode': episode,
            'total_reward': episode_metrics['total_reward'],
            'steps': episode_metrics['steps'],
            'processed_packets': episode_metrics['processed_packets'],
            'rejected_packets': episode_metrics['rejected_packets'],
            'epsilon': agent.epsilon,
            'server_final_states': [{
                'cpu_usage': server.cpu_usage_pct,
                'memory_usage': server.memory_usage_pct,
                'bandwidth_usage': server.bandwidth_usage_pct,
                'available_capacity': server.available_capacity,
                'processed_packets': server.processed_packets
            } for server in train_env.servers]
        })
        
        # Plot and save metrics every 5 episodes
        if episode % 10 == 0:
            plot_training_metrics(all_metrics, episode, output_dirs['plots'])
            save_episode_metrics(all_metrics, episode, output_dirs['metrics'])
            # Save model checkpoint
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon
            }, os.path.join(output_dirs['models'], f'model_checkpoint_episode_{episode}.pth'))
            
        print(f"Episode {episode}/{num_episodes} completed. Total Reward: {episode_metrics['total_reward']:.2f}")
    
    # Save final metrics and model
    save_final_metrics(all_metrics, output_dirs['metrics'])
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, os.path.join(output_dirs['models'], 'final_model.pth'))
    
    return agent, all_metrics

def plot_training_metrics(metrics, episode, plot_dir):
    """Create and save comprehensive training metric plots"""
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Training Rewards with Moving Average
    plt.subplot(3, 2, 1)
    plt.plot(metrics['training_rewards'], alpha=0.6, label='Episode Reward')
    plt.plot(metrics['average_rewards'], color='red', linewidth=2, label='Moving Average')
    plt.title('Training Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot 2: Episode Lengths
    plt.subplot(3, 2, 2)
    plt.plot(metrics['episode_lengths'])
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot 3: Acceptance Rates
    plt.subplot(3, 2, 3)
    plt.plot(metrics['acceptance_rates'])
    plt.title('Packet Acceptance Rates')
    plt.xlabel('Episode')
    plt.ylabel('Acceptance Rate')
    
    # Plot 4: Server Loads
    plt.subplot(3, 2, 4)
    server_loads = np.array(metrics['server_loads'])
    for i in range(server_loads.shape[1]):
        plt.plot(server_loads[:, i], label=f'Server {i+1}')
    plt.title('Server Loads over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Load (%)')
    plt.legend()
    
    # Plot 5: Latest Episode Server Metrics
    plt.subplot(3, 2, 5)
    latest_summary = metrics['episode_summaries'][-1]
    server_metrics = [s['processed_packets'] for s in latest_summary['server_final_states']]
    plt.bar(range(len(server_metrics)), server_metrics)
    plt.title(f'Packet Distribution (Episode {episode})')
    plt.xlabel('Server')
    plt.ylabel('Processed Packets')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'training_metrics_episode_{episode}.png'))
    plt.close()

def save_episode_metrics(metrics, episode, metrics_dir):
    """Save detailed metrics for the current episode"""
    latest_summary = metrics['episode_summaries'][-1]
    
    metrics_file = os.path.join(metrics_dir, f'episode_{episode}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(latest_summary, f, indent=4)

def save_final_metrics(metrics, metrics_dir):
    """Save complete training metrics"""
    final_metrics = {
        'training_summary': {
            'total_episodes': len(metrics['training_rewards']),
            'final_acceptance_rate': metrics['acceptance_rates'][-1],
            'average_reward': np.mean(metrics['training_rewards']),
            'average_episode_length': np.mean(metrics['episode_lengths'])
        },
        'detailed_metrics': metrics
    }
    
    with open(os.path.join(metrics_dir, 'final_training_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)


def evaluate_model(agent, test_env, output_dirs, num_test_episodes=1):
    """
    Evaluate the trained model on unseen test data with enhanced resource tracking
    """
    print("\nStarting Model Evaluation on Test Data...")
    
    test_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'acceptance_rates': [],
        'server_loads': [],
        'episode_summaries': [],
        'latency_distribution': [],
        'step_balance_scores': [],
        'step_server_loads': [],
        'final_loads': [],
        'server_capacities': [],
        # New metrics for resource tracking
        'step_cpu_utilization': [],
        'step_memory_utilization': [],
        'step_bandwidth_utilization': [],
        'step_server_utilization': []  # Overall server utilization score
    }
    
    # Set agent to evaluation mode
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    # Store initial server capacities
    for server in test_env.servers:
        test_metrics['server_capacities'].append({
            'cpu': server.max_cpu,
            'memory': server.max_memory,
            'bandwidth': server.max_bandwidth
        })
    
    # Run one evaluation episode
    state = test_env.reset()
    total_reward = 0
    steps = 0
    processed_packets = 0
    rejected_packets = 0
    
    done = False
    while not done:
        # Get action from trained model
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action_values = agent.q_network(state_tensor)
            action = torch.argmax(action_values).item()
        
        # Take action
        next_state, reward, done, info = test_env.step(action)
        
        # Update counters
        total_reward += reward
        steps += 1
        
        if info['processed']:
            processed_packets += 1
        else:
            rejected_packets += 1
        
        # Track resource utilization for each step
        cpu_utils = []
        memory_utils = []
        bandwidth_utils = []
        server_utils = []
        current_loads = []
        
        for server in test_env.servers:
            # Calculate individual resource utilization
            cpu_util = server.used_cpu / server.max_cpu
            mem_util = server.used_memory / server.max_memory
            bw_util = server.used_bandwidth / server.max_bandwidth
            
            cpu_utils.append(cpu_util * 100)
            memory_utils.append(mem_util * 100)
            bandwidth_utils.append(bw_util * 100)
            
            # Calculate weighted server utilization
            server_util = (cpu_util * 0.4 + mem_util * 0.3 + bw_util * 0.3) * 100
            server_utils.append(server_util)
            current_loads.append(server_util)
        
        # Store utilization metrics for this step
        test_metrics['step_cpu_utilization'].append(cpu_utils)
        test_metrics['step_memory_utilization'].append(memory_utils)
        test_metrics['step_bandwidth_utilization'].append(bandwidth_utils)
        test_metrics['step_server_utilization'].append(server_utils)
        test_metrics['step_server_loads'].append(current_loads)
        
        # Calculate balance score
        if current_loads:
            avg_load = np.mean(current_loads)
            if avg_load > 0:
                std_dev = np.std(current_loads)
                balance_score = 1 - (std_dev / (avg_load + 1e-6))
            else:
                balance_score = 1.0
            test_metrics['step_balance_scores'].append(balance_score)
        
        state = next_state
    
    # Store final metrics
    test_metrics['episode_rewards'].append(total_reward)
    test_metrics['episode_lengths'].append(steps)
    
    if steps > 0:
        test_metrics['acceptance_rates'].append(processed_packets / (processed_packets + rejected_packets))
    
    # Store final loads
    test_metrics['final_loads'] = [utils[-1] for utils in zip(*test_metrics['step_server_utilization'])]
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Save metrics and create plots
    save_test_metrics(test_metrics, output_dirs)
    plot_test_metrics(test_metrics, output_dirs)
    
    return test_metrics

def plot_test_metrics(metrics, output_dirs):
    """Create and save comprehensive test evaluation plots"""
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Load Balance Score Over Steps
    plt.subplot(2, 1, 1)
    if metrics['step_balance_scores']:
        steps = range(len(metrics['step_balance_scores']))
        plt.plot(steps, metrics['step_balance_scores'], 'g-', label='Balance Score')
        plt.title('Load Balance Score Over Steps')
        plt.xlabel('Step (Packet Assignment)')
        plt.ylabel('Balance Score')
        plt.grid(True)
        plt.legend()
    
    # Plot 2: Server Utilization Over Steps
    plt.subplot(2, 1, 2)
    if metrics['step_server_utilization']:
        steps = range(len(metrics['step_server_utilization']))
        for i in range(len(metrics['step_server_utilization'][0])):
            server_utils = [step[i] for step in metrics['step_server_utilization']]
            plt.plot(steps, server_utils, label=f'Server {i+1}')
        plt.title('Server Utilization Over Steps')
        plt.xlabel('Step (Packet Assignment)')
        plt.ylabel('Utilization (%)')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['plots'], 'resource_utilization_metrics.png'))
    plt.close()
    
    # Create separate plot for final load vs capacity
    plt.figure(figsize=(10, 6))
    server_indices = np.arange(len(metrics['final_loads']))
    bar_width = 0.35
    
    # Plot capacity bars (blue)
    capacities = [100] * len(server_indices)
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

def save_test_metrics(metrics, output_dirs):
    """Save test metrics to file with enhanced resource utilization data"""
    test_summary = {
        'total_steps': metrics['episode_lengths'][0] if metrics['episode_lengths'] else 0,
        'total_reward': metrics['episode_rewards'][0] if metrics['episode_rewards'] else 0,
        'acceptance_rate': metrics['acceptance_rates'][0] if metrics['acceptance_rates'] else 0,
        'final_server_loads': metrics['final_loads'],
        'balance_score_progression': metrics['step_balance_scores'],
        'server_utilization': {
            f'server_{i+1}': {
                'mean': np.mean([step[i] for step in metrics['step_server_utilization']]),
                'max': np.max([step[i] for step in metrics['step_server_utilization']]),
                'min': np.min([step[i] for step in metrics['step_server_utilization']])
            } for i in range(len(metrics['step_server_utilization'][0]))
        }
    }
    
    with open(os.path.join(output_dirs['metrics'], 'test_evaluation_metrics.json'), 'w') as f:
        json.dump(test_summary, f, indent=4)

def main():
    # Setup output directories
    output_dirs = setup_output_dirs()
    
    # Load your data
    train_data = pd.read_csv('sampled_20000_data.csv')
    full_test_data = pd.read_csv('network_traffic_subset.csv')
    test_data = full_test_data.sample(n=200, random_state=42)
    
    # Define server configurations with actual resource capacities
    server_configs = [
        {'cpu': 2.0, 'memory': 4, 'bandwidth': 1000},  # 2GHz CPU, 4GB RAM, 1Gbps
        {'cpu': 3.0, 'memory': 8, 'bandwidth': 2000},  # 3GHz CPU, 8GB RAM, 2Gbps
        {'cpu': 1.5, 'memory': 2, 'bandwidth': 500}    # 1.5GHz CPU, 2GB RAM, 500Mbps
    ]
    
    num_episodes = 500
    # Train the agent
    train_env = PacketRoutingEnv(train_data, server_configs)
    test_env = PacketRoutingEnv(test_data, server_configs)
    agent, training_metrics = train_and_evaluate(train_data, test_data, server_configs, output_dirs, num_episodes)
    
    # Evaluate the trained agent
    test_metrics = evaluate_model(agent, test_env, output_dirs)
    
    print("\nEvaluation Results:")
    print(f"Average Test Reward: {np.mean(test_metrics['episode_rewards']):.2f}")
    print(f"Average Acceptance Rate: {np.mean(test_metrics['acceptance_rates']):.2%}")
    print(f"Results saved in: {output_dirs['base']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")