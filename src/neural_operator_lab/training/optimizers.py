"""Optimizers and schedulers for neural operator training."""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, List
import math


def create_optimizer(model_parameters, 
                    optimizer_name: str = 'adamw',
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-5,
                    **kwargs) -> torch.optim.Optimizer:
    """Create optimizer for neural operator training.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    
    if optimizer_name.lower() == 'adam':
        return optim.Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get('nesterov', True)
        )
    
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.0),
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_name.lower() == 'adagrad':
        return optim.Adagrad(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            lr_decay=kwargs.get('lr_decay', 0),
            eps=kwargs.get('eps', 1e-10)
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_name: str = 'cosine',
                    **kwargs) -> Optional[_LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler ('cosine', 'step', 'plateau', 'exponential', 'linear')
        **kwargs: Scheduler-specific arguments
        
    Returns:
        Configured scheduler or None
    """
    
    if scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_name.lower() == 'cosine_warm':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 0),
            warmup_steps=kwargs.get('warmup_steps', 0)
        )
    
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_name.lower() == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [50, 80]),
            gamma=kwargs.get('gamma', 0.1),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 0)
        )
    
    elif scheduler_name.lower() == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_name.lower() == 'linear':
        return LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 1.0),
            end_factor=kwargs.get('end_factor', 0.1),
            total_iters=kwargs.get('total_iters', 100)
        )
    
    elif scheduler_name.lower() == 'warmup':
        return WarmupLR(
            optimizer,
            warmup_steps=kwargs.get('warmup_steps', 1000),
            max_lr=kwargs.get('max_lr', optimizer.param_groups[0]['lr']),
            min_lr=kwargs.get('min_lr', 0)
        )
    
    elif scheduler_name.lower() == 'none':
        return None
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class LinearLR(_LRScheduler):
    """Linear learning rate scheduler."""
    
    def __init__(self, optimizer, start_factor=1.0, end_factor=0.1, total_iters=100, last_epoch=-1):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            factor = self.end_factor
        else:
            factor = self.start_factor + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
        
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupLR(_LRScheduler):
    """Warmup learning rate scheduler."""
    
    def __init__(self, optimizer, warmup_steps=1000, max_lr=None, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr or optimizer.param_groups[0]['lr']
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            factor = self.last_epoch / self.warmup_steps
            return [self.min_lr + (self.max_lr - self.min_lr) * factor for _ in self.base_lrs]
        else:
            return [self.max_lr for _ in self.base_lrs]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine annealing with warm restarts and optional warmup."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, warmup_steps=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.T_i = T_0
        self.T_cur = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            factor = self.last_epoch / self.warmup_steps
            return [self.eta_min + (base_lr - self.eta_min) * factor for base_lr in self.base_lrs]
        
        # Adjust for warmup
        adjusted_epoch = self.last_epoch - self.warmup_steps
        
        # Find current cycle
        if adjusted_epoch >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        else:
            self.T_cur = adjusted_epoch
        
        # Cosine annealing
        factor = (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        return [self.eta_min + (base_lr - self.eta_min) * factor for base_lr in self.base_lrs]


class AdaBound(torch.optim.Optimizer):
    """AdaBound optimizer combining Adam and SGD."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))

        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsbound']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsbound']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsbound']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Apply bias correction
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / defaults['lr']
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(step_size, alpha=-1)

        return loss


def get_optimizer_config(pde_type: str) -> Dict[str, Any]:
    """Get recommended optimizer configuration for different PDE types.
    
    Args:
        pde_type: Type of PDE ('navier_stokes', 'heat', 'wave', 'burgers', 'darcy')
        
    Returns:
        Dictionary with optimizer and scheduler configuration
    """
    
    configs = {
        'navier_stokes': {
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'scheduler': 'cosine_warm',
            'T_0': 20,
            'warmup_steps': 500
        },
        
        'heat': {
            'optimizer': 'adam',
            'learning_rate': 5e-4,
            'weight_decay': 1e-6,
            'scheduler': 'step',
            'step_size': 30,
            'gamma': 0.5
        },
        
        'wave': {
            'optimizer': 'adamw',
            'learning_rate': 2e-4,
            'weight_decay': 1e-5,
            'scheduler': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6
        },
        
        'burgers': {
            'optimizer': 'adam',
            'learning_rate': 1e-3,
            'weight_decay': 0,
            'scheduler': 'plateau',
            'patience': 10,
            'factor': 0.5
        },
        
        'darcy': {
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'scheduler': 'multistep',
            'milestones': [50, 80],
            'gamma': 0.1
        }
    }
    
    return configs.get(pde_type, configs['navier_stokes'])


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer."""
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """First step: compute and apply the perturbation."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Second step: restore original parameters and apply base optimizer."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # restore original parameters

        self.base_optimizer.step()  # apply the base optimizer step

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Combined step that performs both first and second steps."""
        assert closure is not None, "SAM requires a closure function."
        
        # First step
        self.first_step(zero_grad=True)
        
        # Re-evaluate the function at the perturbed point
        closure()
        
        # Second step
        self.second_step()

    def _grad_norm(self):
        """Compute the gradient norm."""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            dtype=torch.float32
        )
        return norm.to(shared_device)