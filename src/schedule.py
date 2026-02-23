from typing import Optional
import torch
from torch._C._distributed_c10d import Work

from comms import PipelineComms
from model import ShardedMLP


def naive_pipeline_step(model: ShardedMLP, comms: PipelineComms, batch:torch.Tensor, targets:torch.Tensor, hidden_dim:int, device):
    """
    A single training step using the Naive (Stop-and-Wait) schedule.
    """

    # --- PHASE 1: FORWARD PASS ---

    # A. Get Input
    if comms.rank == 0:
        # Rank 0 gets the data directly from the dataloader
        input_data = batch
    else:
        # we need the shape here, but not for above because
        # the first device just gets the data straight from
        # the data loader and doesn't need to make a buffer
        # tensor to receive the activations
        shape = (batch, hidden_dim)
        # Others wait to receive from the left
        input_data = comms.recv_forward(shape, device)
        # When we receive a tensor via dist.recv, it is a leaf node
        # By default, leaf nodes have requires_grad = False.
        # Because we set requires_grad = True, the engine
        # continues the chain rule all the way back to that input tensor
        # Without this, the tensor would be treated as a constant by autograd and
        # input_data.grad would be set to None in backward(),
        # so earlier (upstream) model parameters wouldn’t update
        input_data.requires_grad = True # NOTE: requires_grad=True, 使得input_data的梯度可以被计算出来

    # B. Compute
    # if you are not last, you just calculate activations
    # if you are last, you also calculate the loss with targets, which is output
    output = model.forward(input_data, targets)

    # C. Send data to the right, ie: send activations to the next rank
    if not model.is_last:
        # NOTE: 注意，此处output.detach()会使detach()后的数据没有梯度, 即发送到下一个rank的数据里没有grad, 
        # 如需梯度，需要在receive后手动设置requires_grad=True
        comms.send_forward(output.detach()) 

    # --- PHASE 2: BACKWARD PASS ---

    # A. Get Gradients
    if model.is_last:
        loss = output
        # Scalar Backward: loss.backward() (Used only on the last GPU).
        # Non-Scalar Backward: output.backward(gradient) (Used on all previous GPUs).
        loss.backward()  # This starts the chain reaction, 只累加梯度
    else:
        # Receive gradients coming from the right
        # whereas in the forward pass, we don't have the batch size unless we are the first device,
        # since gradients are propagated for every activation computed, we can just take
        # the activation dimensions for the shape of the gradient tensor we receive in backward()
        # 梯度必须与output.shape一致，因为backward()会自动计算梯度
        grad_from_next = comms.recv_backward(output.shape, device)
        # B. Compute Local Gradients
        # This is the "Backprop" step connecting the received grad to our weights
        # When you call .backward() on a non-scalar tensor (like a hidden activation with
        # shape [32, 128]), PyTorch requires a "matching" gradient tensor of the same shape.
        # This provided gradient acts as the starting point for the Vector-Jacobian Product,
        # allowing the chain rule to flow backward to the weights and the input.
        output.backward(grad_from_next)

    # 得到input_data的梯度
    grad_to_send = input_data.grad
    """
    ∂Weights/∂Loss are the gradients which tell the model how to change its own internal layers.
    ∂Input/∂Loss are the gradients which we back-propagate; if Rank 0 is the very first layer
    (taking in the raw data/images), it technically calculates the gradient with respect to
    the raw input, but we discard this because we can't "update" the training data!
    """
    # C. Send Gradients
    if not model.is_first:
        # 向前面的层反向传播梯度
        comms.send_backward(grad_to_send)

    if model.is_last:
        return loss


def gpipe_pipeline_step(model, comms:PipelineComms, batch:torch.Tensor, targets, hidden_dim, chunk_num, device):
    """
    将数据分为micro-batch, 即先将所有的数据进行完全的前向，然后再一起进行反向, 好处是可以减少空泡率

    GPipe Schedule: FWD all chunks -> BWD all chunks.
    """
    # 1. Prepare Data Slices
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunk_num)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunk_num)

    # Storage for "Phase 2"
    input_buffers = []
    output_buffers = []

    """
    转变思维，注意是在当前Rank上，不需要for循环
    """
    # --- PHASE 1: ALL FORWARDS (Fill the Pipe) ---
    for i in range(chunk_num):
        # A. Setup Input
        if comms.rank == 0:
            input_data = micro_batches[i]
        else:
            shape = (batch // chunk_num, hidden_dim)
            # NOTE: 注意，这里也是同步等待
            input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True

        # B. Forward Pass
        if comms.rank == comms.world_size - 1:
            output = model(input_data, micro_targets[i])
        else:
            # 如果不是最后一个rank, 不需要传输target
            output = model(input_data)
            comms.send_forward(output.detach())

        # D. Buffer for Backward
        input_buffers.append(input_data)
        output_buffers.append(output)  # On last rank, this is the Loss
    # END FOR

    # --- PHASE 2: ALL BACKWARDS (Drain the Pipe) ---
    if comms.rank == comms.world_size - 1:
        # total_loss: scalar
        total_loss = torch.zeros(output.shape, device=device)

    # Layers: Reverse Order (handled by Autograd).
    # Micro-batches: Forward Order (handled by loop to match the send/recv order).
    # Think of a conveyor belt
    # Both loop orders give the same result here because each micro-batch's
    # forward and backward passes are fully independent of the others in this
    # GPipe schedule: all forwards are completed and stored before any backward
    # begins, so the order of backward iteration (reversed or not) does not
    # change gradients or loss accumulation across chunks.
    for i in range(chunk_num):
        # Retrieve state from Phase 1
        input_data = input_buffers[i]
        output = output_buffers[i]

        if comms.rank == comms.world_size - 1:
            # On Last Rank, 'output' IS the loss
            loss = output / chunk_num # 每个只累加 1/chunk_num的梯度
            # NOTE: 注意，梯度反向传播后，会自动释放activations所占用的GPU缓存
            loss.backward() # loss.backward()会自动累加梯度， 但没有更新weights, 也不会清空梯度
            total_loss += loss
        else:
            # On other ranks, we need gradients from downstream
            # NOTE:接收下一层的梯度
            grad_from_next = comms.recv_backward(output.shape, device)
            output.backward(grad_from_next)

        # Send gradients backward (if not first)
        if comms.rank != 0:
            # 向前面的层传播梯度
            comms.send_backward(input_data.grad)

    # Return loss across chunks (for logging) if last rank
    if comms.rank == comms.world_size - 1:
        return total_loss


def one_forward_one_backward_pipeline_step(model, comms:PipelineComms, batch:torch.Tensor, targets, hidden_dim:int, chunk_num:int, device):
    """
    1F1B Schedule: Interleaves Forward and Backward passes.
    交错进行前向和反向, 好处是即时释放activation，减少GPU内存peak memory的占用
    """
    # 1. Prepare Data Slices
    if comms.rank == 0:
        # 在batch维度将batch切分成chunks个micro_batches, 每个micro_batch的shape为(batch/chunks, hidden_dim)
        micro_batches = torch.chunk(batch, chunk_num, dim=0)
    if comms.rank == comms.world_size - 1: # 最后一个GPU, 将targets切分成chunks个micro_targets, 每个micro_target的shape为(batch/chunks)
        micro_targets = targets.chunk(chunk_num)

    # Storage for "Phase 2"
    # NOTE:对于每个rank上的每个micro_batch的数据，都需要保存每个micro_batch_idx 的input_data和output
    input_buffers = [None] * chunk_num # 每个micro_batch的输入数据
    output_buffers = [None] * chunk_num # 每个micro_batch的输出数据, 即激活值或者loss
    async_requests = []  # Keep request objects alive to prevent buffer deallocation, 异步通信均需要将request对象保存在列表中，以防止缓冲区被释放

    """
    注意：这里的warmup和cooldown的逻辑，是针对各rank上的gpu流水线是否饱和而言的，
    不是指lr中的warmup
    """
    # Schedule Logic
    # Rank 0 (First) has max warmup (needs to fill the whole pipe)
    # Rank N (Last) has 0 warmup (can backward immediately)
    num_warmup = comms.world_size - comms.rank - 1  # 物理意义：即当前rank所在gpu后面的gpu的数量, 后面有多少个gpu, 则warmup几次
    num_1f1b = chunk_num - num_warmup # chunk_num中减去num_warmup的数量，即可得到1f1b的chunk数量
    # 同时，cooldown的次数与num_warmup的数量相同

    """
    若world_size=4, chunk_num=8, 所有的micro_batch总共会有8*2=16个前向+反向
        rank=0, 则num_warmup=4-0-1=3, num_1f1b=8-3=5
        rank=1, 则num_warmup=4-1-1=2, num_1f1b=8-2=6
        rank=2, 则num_warmup=4-2-1=1, num_1f1b=8-1=7
        rank=3, 则num_warmup=4-3-1=0, num_1f1b=8-0=8
    
    异步通信的原因分析
    1. 性能优化策略
    - 前向传播的异步发送：在1F1B调度中，前向传播阶段可以异步发送激活值给下一层，因为下一层可以立即开始计算，不需要等待当前层完成所有操作
    - 反向传播的同步通信：梯度传播需要严格的顺序依赖，必须等待前一层的梯度计算完成才能继续
    2. 数据依赖关系
    - 前向传播：激活值发送后，当前层可以继续处理下一个micro-batch，与下一层的计算可以并行
    - 反向传播：梯度必须按顺序传播，因为每个层的梯度计算依赖于后一层传回的梯度
    3. 代码中的具体实现
        1 # 异步发送前向激活值
        2 req: Optional[Work] = comms.isend_forward(output.detach())
        3 
        4 # 同步接收前向数据） 
        5 input_data = comms.recv_forward(shape, device)
        6 
        7 # 同步接收反向梯度
        8 grad_from_next = comms.recv_backward(output.shape, device)
        9 
        10 # 同步发送反向梯度
        11 comms.send_backward(input_data.grad)
    4. 设计原理
    - 最大化流水线利用率：通过异步发送前向激活值，可以让计算和通信重叠，减少流水线中的气泡（bubble）
    - 保证正确性：反向传播必须同步，确保梯度计算的正确顺序和依赖关系
    这种混合通信模式是1F1B调度算法的核心优化，能够在保证正确性的同时最大化并行效率。
    """

    # 仅对一个micro_batch进行前向传播
    def run_forward(micro_batch_idx:int):
        # ... Setup Input ...
        if comms.rank == 0:
            input_data = micro_batches[micro_batch_idx]
        else:
            shape = (batch // chunk_num, hidden_dim)
            # 同步的等待前面的layer的forward输出
            input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True

        # B. Forward Pass
        if comms.rank == comms.world_size - 1:
            output = model(input_data, micro_targets[micro_batch_idx])
        else:
            output = model(input_data)
            # ASYNC SEND - returns immediately, doesn't block
            # 
            # NOTE: 异步发送前向数据至下一层layer
            # 前向传播的异步发送：结合pdf中的调度图，在1F1B调度图中，前向传播阶段可以异步发送激活值给下一层, 
            # rank2向rank3发送mb2的前向激活值,rank3向rank2发送mb1的反向梯度， 因此rank2,rank3之间有死锁，
            # 即两者都同时向对方发送, 不同的数据，都要对方同步式接收，而大家都是阻塞式发送，根本没有空闲去接收数据， 
            # 导致产生死锁，所以需要异步发送
            # NOTE:各rank之间的前向发送是异步的，但各rank内的各micro_batch的的前向与反向是同步的，所以并不会产生数据错误，即使用的是旧数据
            req: Optional[Work] = comms.isend_forward(output.detach()) # 返回异步请求对象
            async_requests.append(req)  # Keep request alive to prevent buffer deallocation

        input_buffers[micro_batch_idx] = input_data
        output_buffers[micro_batch_idx] = output

    # 仅对一个micro_batch进行反向传播
    def run_backward(micro_batch_idx:int):
        input_data = input_buffers[micro_batch_idx]
        output = output_buffers[micro_batch_idx]

        if comms.rank == comms.world_size - 1:
            # NOTE: loss为何要除以chunk_num? 因为loss本身是在batch维度平均的，
            # 然后在累加chunk_num次微批次数据的梯度，所以此处除以chunk_num以恢复batch维度的平均
            loss = output / chunk_num 
            loss.backward() # NOTE: backward()会自动累加梯度,自动计算input_data.grad，但没有更新weights, 直到调用optimizer.step()才会更新weights, 一共累加了chunk_num次微批次数据的梯度
        else:
            grad_from_next: torch.Tensor = comms.recv_backward(output.shape, device)
            output.backward(grad_from_next)

        if comms.rank != 0:
            comms.send_backward(input_data.grad)

        if comms.rank == comms.world_size - 1:
            return loss

    # --- EXECUTION PHASES ---
    if comms.rank == comms.world_size - 1:
        total_loss: torch.Tensor = torch.zeros(1, device=device)

    """
    world_size=4, chunk_num=8
    注意：在当前rank的GPU上进行前向与反向, 请参照1F1B Schedule的流水线图
    rank=0: num_warmup=3次前向 + num_1f1b=5次前向和反向 + num_warmup=3次反向
    对每个micro-batch展开有：
        - 3次前向：m0-f, m1-f, m2-f, 注意：前向与反向之间还有一些空泡时间，未列出
        - 5次前向和反向：(m3-f,m0-b), (m4-f,m1-b), (m5-f,m2-b), (m6-f,m3-b), (m7-f,m4-b), 前向与反向micro_batch的index相差3
        - 3次反向：m5-b, m6-b, m7-b, 注意：前向与反向index之间相差5

    rank=1: num_warmup=2次前向 + num_1f1b=6次前向和反向 + num_warmup=2次反向
    对每个micro-batch展开有：
        - 2次前向：m0-f, m1-f
        - 6次前向和反向：(m2-f,m0-b), (m3-f,m1-b), (m4-f,m2-b), (m5-f,m3-b), (m6-f,m4-b), (m7-f,m5-b), 前向与反向micro_batch的index相差2
        - 2次反向：m6-b, m7-b, 注意：前向与反向index之间相差6

    rank=2: num_warmup=1次前向 + num_1f1b=7次前向和反向 + num_warmup=1次反向
    对每个micro-batch展开有：
        - 1次前向：m0-f
        - 7次前向和反向：(m1-f, m0-b), (m2-f,m1-b), (m3-f,m2-b), (m4-f,m3-b), (m5-f,m4-b), (m6-f,m5-b), (m7-f,m6-b), 前向与反向micro_batch的index相差1
        - 1次反向：m7-b, 注意：前向与反向index之间相差7

    rank=3: num_warmup=0次前向 + num_1f1b=8次前向和反向 + num_warmup=0次反向
    对每个micro-batch展开有：
        - 0次前向：
        - 6次前向和反向：(m0-f,m0-b), (m1-f,m1-b), (m2-f,m2-b), (m3-f,m3-b), (m4-f,m4-b), (m5-f,m5-b), (m6-f,m6-b), (m7-f,m7-b), 前向与反向micro_batch的index相差0
        - 0次反向：

    """
    # 必须按1F1B Schedule的顺序进行各micro_batch的前向与反向，否则会导致数据错误
    # Phase 1: Warmup (Forward Only), 第一阶段只进行前向传播，打满流水线
    for i in range(num_warmup):
        run_forward(micro_batch_idx=i)

    # Phase 2: Steady State (1F1B), 第二阶段进行交错的前向和反向传播
    for i in range(num_1f1b):
        # 一次前向传播后马上进行一次反向传播
        run_forward(micro_batch_idx=i + num_warmup)
        # run_backward returns the loss (on last rank) or None (others)
        res = run_backward(micro_batch_idx=i)
        if comms.rank == comms.world_size - 1:
            total_loss += res

    # Phase 3: Cooldown (Backward Only), 注意：backward only阶段只在非末位rank上进行,此时loss已经累加完成，此处无需累加loss
    for i in range(num_warmup):
        run_backward(micro_batch_idx=i + num_1f1b)

    # Return Loss
    return total_loss if comms.rank == comms.world_size - 1 else None