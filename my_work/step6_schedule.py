import torch
from step2_comms import PipelineComms
from step4_model import ShardedMLP

def naive_pipeline_step(model: ShardedMLP, comms: PipelineComms, batch, targets, hidden_dim, device):
    """
    A single training step using the Naive (Stop-and-Wait) schedule.

    TODOs:
    - Receive input from previous stage if not first stage (requires_grad)
    - Forward batch through model
    - Send output to next stage if not last stage (detach)
    - Perform backward pass: 
        - If last stage, compute loss and call backward on it
        - Else, receive grad from next stage and call backward
    - Send grad to previous stage if not first stage
    - Return loss if last stage, else None
    """
    # TODO: If comms.rank == 0, use 'batch' directly; else, receive input
    # TODO: Forward pass through model
    # TODO: If not last stage, send output to next stage
    # TODO: Backward pass (different for last and non-last stage)
    # TODO: Send grad to previous stage if not first
    # TODO: Return loss if last stage, else None
    pass

def gpipe_pipeline_step(model, comms, batch, targets, hidden_dim, chunks, device):
    """
    GPipe Schedule: FWD all chunks -> BWD all chunks.
    """
    # 1. Prepare Data Slices
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunks)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunks)
    
    # Storage for "Phase 2"
    input_buffers = [] 
    output_buffers = []
    
    # --- PHASE 1: ALL FORWARDS (Fill the Pipe) ---
    for i in range(chunks):
        # A. Setup Input
        if comms.rank == 0:
            input_data = micro_batches[i]
        else:
            shape = (batch//chunks, hidden_dim)
            input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True

        # B. Forward Pass
        if comms.rank == comms.world_size - 1:
            output = model(input_data, micro_targets[i])
        else:
            output = model(input_data)
            comms.send_forward(output.detach())

        # D. Buffer for Backward
        input_buffers.append(input_data)
        output_buffers.append(output) # On last rank, this is the Loss

    # --- PHASE 2: ALL BACKWARDS (Drain the Pipe) ---
    if comms.rank == comms.world_size - 1:
        total_loss = torch.zeros(output.shape)
    # Layers: Reverse Order (handled by Autograd).
    # Micro-batches: Forward Order (handled by loop to match the send/recv order).
    # Think of a conveyor belt
    # Both loop orders give the same result here because each micro-batch's 
    # forward and backward passes are fully independent of the others in this 
    # GPipe schedule: all forwards are completed and stored before any backward 
    # begins, so the order of backward iteration (reversed or not) does not 
    # change gradients or loss accumulation across chunks.
    for i in range(chunks):
        # Retrieve state from Phase 1
        input_data = input_buffers[i]
        output = output_buffers[i]
        
        if comms.rank == comms.world_size - 1:
            # On Last Rank, 'output' IS the loss
            loss = output / chunks
            loss.backward()
            total_loss += loss
        else:
            # On other ranks, we need gradients from downstream
            grad_from_next = comms.recv_backward(output.shape, device)
            output.backward(grad_from_next)
            
        # Send gradients backward (if not first)
        if comms.rank != 0:
            comms.send_backward(input_data.grad)
            
    # Return loss across chunks (for logging) if last rank
    if comms.rank == comms.world_size - 1:
        return total_loss

def onef_oneb_pipeline_step(model, comms, batch, targets, hidden_dim, chunks, device):
    """
    1F1B Schedule: Interleaves Forward and Backward passes in a pipelined manner.
    """
    # TODO: Chunk the batches into microbatches and the targets in to microtargets
    # TODO: Initialize buffers for activations, gradients, etc.

    # Forward warmup: Fill the pipeline
    # for i in range(num_warmup_steps):
    #     - If comms.rank == 0, use microbatch directly; else, receive input
    #     - Forward microbatch through model
    #     - If not last stage, send output to next stage
    #     - Append input/output to buffers

    # 1F1B Steady State
    # for i in range(num_steady_steps):
    #     - Forward pass for new microbatch (as above)
    #     - Backward pass for previous microbatch
    #         - If last stage, compute loss and call backward
    #         - Else, receive grad from next stage and call backward
    #         - Send grad to previous stage if not first

    # Backward drain: Complete outstanding backward passes
    # for i in range(num_drain_steps):
    #     - Backward pass for remaining microbatches (as above)

    # TODO: Return loss if last stage, else None
    # 1. Prepare Data Slices
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunks)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunks)
    
    input_buffers = [None] * chunks 
    output_buffers = [None] * chunks 
    async_requests = []
    
    warmup = comms.world_size - comms.rank - 1
    onef_oneb = chunks - warmup
    
    def forward(micro_batch_idx):
        if comms.rank == 0:
            input_data = micro_batches[micro_batch_idx]
        else:
            shape = (batch//chunks,hidden_dim)
            input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True

        if comms.rank == comms.world_size - 1:
            output = model(input_data, micro_targets[micro_batch_idx])
        else:
            output = model(input_data)
            req = comms.isend_forward(output.detach())
            async_requests.append(req)

        input_buffers[micro_batch_idx] = input_data
        output_buffers[micro_batch_idx] = output

    def backward(micro_batch_idx):
        input_data = input_buffers[micro_batch_idx]
        output = output_buffers[micro_batch_idx]

        if comms.rank == comms.rank == comms.world_size - 1:
            loss = output / chunks
            loss.backward()
        else:
            grads = comms.recv_backward(output.shape, device)
            torch.autograd.backward(output, grads)
        if comms.rank != 0:
            comms.send_backward(input_data.grad)
        if comms.rank == comms.rank == comms.world_size - 1:
            return loss
        
    if comms.rank == comms.world_size - 1:
        total_loss = torch.zeros(1, device=device)

    # Phase 1: Warmup (Forward Only)
    for i in range(warmup):
        forward(i)

    # Phase 2: Steady State (1F1B)
    for i in range(onef_oneb):
        forward(i + warmup)
        # run_backward returns the loss (on last rank) or None (others)
        res = backward(i)
        if comms.rank == comms.world_size - 1:
            total_loss += res

    # Phase 3: Cooldown (Backward Only)
    for i in range(warmup):
        backward(i + onef_oneb)
    
    # Return Loss
    return total_loss if comms.rank == comms.world_size - 1 else None