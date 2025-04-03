#!/usr/bin/env python3
import torch
import torch.nn as nn

from memory_profiler import MemoryTracer

class MLP(nn.Module):
    """A minimal MLP layer for testing."""

    def __init__(self, hidden_size):
        super().__init__()
        self.fc_1 = nn.Linear(hidden_size, hidden_size*2, bias=False)
        self.fc_2 = nn.Linear(hidden_size*2, hidden_size, bias=False)

    def forward(self, x):
        x = self.fc_1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc_2(x)
        return x

class SimpleModel(nn.Module):
    """A minimal model for testing."""

    def __init__(self, hidden_size, num_layers, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [MLP(hidden_size) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

batch_size = 1
seq_length = 16
hidden_size = 1024
num_layers = 16
vocab_size = 1000
device = 'cuda'
dtype = torch.float16

def run_estimation():
    """Run memory estimation test with a minimal model."""
    estimator = MemoryTracer(device=device)
    torch.set_default_device('cuda:0')

    # Use fake tensors and track memory
    with estimator:
        # Create model
        model = SimpleModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
        ).cuda()

        # Register hooks
        hook_handles = estimator.memory_dispatch_mode.register_hooks_to_module(model)
        model = model.to(dtype=torch.float32)

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create fake input
        fake_input = estimator.create_fake_tensor(
            batch_size, seq_length, 1,
            dtype=torch.long,
            device=device,
        ).squeeze(-1)

        # Loss function
        loss_fn = nn.CrossEntropyLoss().to(device)

        # Simulate one training iteration
        optimizer.zero_grad()

        # Forward pass
        logits = model(fake_input)
        target = fake_input.clone().view(-1)
        logits = logits.view(-1, vocab_size)
        loss = loss_fn(logits, target)
        print(f"profiled current_memory_allocated after forward pass: {estimator.get_current_memory_allocated()[0] / (1024 ** 2):.2f} MB")

        # Backward pass
        loss.backward()
        print(f"profiled current_memory_allocated after backward pass: {estimator.get_current_memory_allocated()[0] / (1024 ** 2):.2f} MB")
        # Optimizer step
        optimizer.step()
        print(f"profiled current_memory_allocated after optimizer step: {estimator.get_current_memory_allocated()[0] / (1024 ** 2):.2f} MB")

        # Remove hooks
        estimator.memory_dispatch_mode.remove_hooks(hook_handles)

    # Print memory statistics
    # print("\n===== MEMORY USAGE =====")
    # estimator.print_memory_stats(detailed=True)


def run_actual():
    torch.set_default_device('cuda:0')

    # Create model
    model = SimpleModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
    ).cuda()

    model = model.to(dtype=torch.float32)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create fake input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Simulate one training iteration
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(input_ids)
    target = input_ids.clone().view(-1)
    logits = logits.view(-1, vocab_size)
    loss = loss_fn(logits, target)
    # print(f"actual max_memory_allocated after forward pass: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"actual current_memory_allocated after forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    
    # Backward pass
    loss.backward()
    print(f"actual current_memory_allocated after backward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

    # Optimizer step
    optimizer.step()
    print(f"actual current_memory_allocated after optimizer step: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    run_estimation()
    torch.cuda.reset_peak_memory_stats() 
    run_actual()
