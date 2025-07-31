import torch
import torch.nn as nn
import numpy as np

class TD3LagrangianController:
    """
    Simple and reliable Lagrangian controller that avoids all compilation issues
    """
    def __init__(self, cost_threshold=0.0, kp=0.05, ki=0.0005, kd=0.1, 
                 lambda_lr=0.01, lambda_max=100.0, device=None):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        
        # Store everything as Python primitives to avoid tensor compilation issues
        self.cost_threshold = float(cost_threshold)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
        
        # State as Python floats
        self.lambda_multiplier_val = 0.0
        self.integral_term = 0.0
        self.prev_cost = 0.0
        self.cost_history = []
        
    def update(self, current_cost):
        """Pure Python implementation - no tensor operations inside"""
        # Convert to Python float
        if torch.is_tensor(current_cost):
            current_cost = current_cost.item()
            
        # PID calculation in pure Python
        error = current_cost - self.cost_threshold
        self.integral_term = 0.95 * self.integral_term + error
        derivative = current_cost - self.prev_cost
        
        pid_signal = (self.kp * error + 
                     self.ki * self.integral_term + 
                     self.kd * max(0.0, derivative))
        
        # Update lambda
        self.lambda_multiplier_val = max(0.0, min(
            self.lambda_multiplier_val + self.lambda_lr * pid_signal,
            self.lambda_max
        ))
        
        self.prev_cost = current_cost
        self.cost_history.append(current_cost)
        
        if len(self.cost_history) > 1000:
            self.cost_history.pop(0)
            
        # Return as tensor on correct device
        return torch.tensor(self.lambda_multiplier_val, device=self.device, dtype=torch.float32)
    
    def get_logs(self):
        return {
            'lagrangian_multiplier': self.lambda_multiplier_val,
            'integral_term': self.integral_term,
            'current_cost': self.prev_cost,
            'cost_threshold': self.cost_threshold,
            'constraint_violation': max(0, self.prev_cost - self.cost_threshold),
            'avg_cost_last_100': np.mean(self.cost_history[-100:]) if len(self.cost_history) > 0 else 0.0
        }
