"""
Advanced optimizers: SAM (Sharpness-Aware Minimization) and Lookahead.

References:
- SAM: "Sharpness-Aware Minimization for Efficiently Improving Generalization" (Foret et al., 2020)
- Lookahead: "Lookahead Optimizer: k steps forward, 1 step back" (Zhang et al., 2019)
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SAM(optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    
    Wraps an existing optimizer to perform SAM updates.
    """
    
    def __init__(self, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(base_optimizer.param_groups, defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.load_state_dict(state_dict)


class Lookahead(optim.Optimizer):
    """
    Lookahead optimizer wrapper.
    
    Wraps an existing optimizer to perform lookahead updates.
    """
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(base_optimizer.defaults)
        self.defaults.update(defaults)
        self.state = base_optimizer.state
        self.alpha = alpha
        self.k = k
        
        # Cache slow weights
        for group in self.param_groups:
            group["slow_params"] = [p.clone().detach() for p in group["params"]]
    
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        
        for group in self.param_groups:
            if self.state[group["params"][0]]["lookahead_step"] >= self.k:
                self.state[group["params"][0]]["lookahead_step"] = 0
                
                # Update slow weights
                for p, slow_p in zip(group["params"], group["slow_params"]):
                    slow_p.add_(p - slow_p, alpha=self.alpha)
                    p.copy_(slow_p)
            else:
                self.state[group["params"][0]]["lookahead_step"] += 1
        
        return loss
    
    def state_dict(self):
        state_dict = self.base_optimizer.state_dict()
        state_dict["lookahead"] = {
            "slow_params": [
                [p.clone().detach() for p in group["slow_params"]]
                for group in self.param_groups
            ]
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        if "lookahead" in state_dict:
            for group, slow_params_group in zip(self.param_groups, state_dict["lookahead"]["slow_params"]):
                group["slow_params"] = [p.clone().detach() for p in slow_params_group]

