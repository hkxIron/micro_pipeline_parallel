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
    # 这是整个任务全局对象，因此只有任务结束后，才会释放，所以可以用来跟踪每个micro_batch的通信状态
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
            """
            上面是一段1F1B的代码，解释一下为何需要异步发送isend_forward，另外，isend_forward如何保证rank之间发送与接收的数据不会产生micro_batch_idx错乱？
            
            这段 1F1B（One Forward One Backward）的代码写得非常标准，注释也非常精彩！你提出的两个问题直击流水线并行（Pipeline Parallelism）中最核心的**死锁（Deadlock）**和**时序（Synchronization）**问题。

            我们分两部分来详细解答：

            ---

            ### 问题一：为什么必须使用异步发送 `isend_forward`？

            在之前的 GPipe 代码中，前向和反向是完全分开的（先做完所有 Forward，再做所有 Backward），所以用同步的 `send` 没问题。但在 1F1B 调度中，**如果不使用异步发送，必然会导致死锁（Deadlock）。**

            我们来看 1F1B 的**第二阶段（Steady State）**，此时前向和反向是交错进行的。假设我们有 Rank 1 和 Rank 2，它们进入了 1F1B 阶段：

            **如果全部使用同步 `send`，会发生什么？**
            1. **Rank 1** 执行 `run_forward`，计算完后，调用同步的 `dist.send(前向激活值)` 给 Rank 2。
            * *注意：同步 `send` 会阻塞，直到 Rank 2 调用了 `recv` 接收这些数据，Rank 1 才能往下走。*
            2. **与此同时，Rank 2** 刚好执行完了 `run_backward`，它需要把梯度传回给 Rank 1，于是调用同步的 `dist.send(反向梯度)` 给 Rank 1。
            * *注意：同步 `send` 也会阻塞，直到 Rank 1 调用了 `recv` 接收这些梯度，Rank 2 才能往下走。*

            **死锁诞生了：**
            * Rank 1 举着“前向激活值”，死死等待 Rank 2 来接收。
            * Rank 2 举着“反向梯度”，死死等待 Rank 1 来接收。
            * 双方都在等对方先腾出手来（调用 `recv`），结果谁也无法执行到下一行代码，整个训练程序卡死。

            **`isend_forward` 如何破局？**
            使用 `isend`（异步发送）打破了这个循环等待：
            1. **Rank 1** 计算完前向后，调用 `dist.isend(前向激活值)`。这相当于 Rank 1 把数据交给了底层的快递员（NCCL/网络硬件），**然后立刻返回，不等待 Rank 2 接收**。
            2. Rank 1 马上往下执行，进入 `run_backward`，并调用 `recv_backward`。
            3. 此时，Rank 2 正在调用 `send_backward` 发送梯度。因为 Rank 1 已经准备好 `recv_backward` 了，所以反向梯度的通信顺利完成！
            4. 随后，Rank 2 进入下一个循环的 `run_forward`，调用 `recv_forward`，正好接住了刚才 Rank 1 异步发过来的前向激活值。

            **总结：** 异步发送前向数据，是为了让当前 GPU 的 CPU 线程不被阻塞，能够赶紧往下走去接收反向梯度，从而**打破交叉通信带来的死锁**。同时，这也实现了通信与计算的重叠（Overlap），提升了性能。

            ---

            ### 问题二：`isend_forward` 如何保证 `micro_batch_idx` 不会错乱？

            你可能会担心：既然是异步发送，CPU 瞬间把好几个 micro_batch 的发送指令都扔出去了，网络会不会把顺序搞乱？Rank 2 接收时会不会把 MB_1 当成了 MB_0？

            答案是：**绝对不会。底层网络严格保证了 FIFO（先进先出）。**

            这与之前 GPipe 的原理是一脉相承的，即使变成了异步（Async），FIFO 的铁律依然生效：

            #### 1. 异步只是 CPU 不等，但网络队列是严格排队的
            当你调用 `dist.isend(tensor)` 时，PyTorch 只是把这个发送任务**按顺序**压入了底层通信引擎（如 NCCL）的指令队列中。
            * 第一次调用 `run_forward(0)` -> 压入发送 MB_0 的指令。
            * 第二次调用 `run_forward(1)` -> 压入发送 MB_1 的指令。

            NCCL 在处理这两个 Rank 之间的点对点通信时，是一条单行道。**先压入队列的指令，必定先在物理网络中传输；先在网络中传输的数据，必定先到达接收端。**

            #### 2. 发送端和接收端的循环顺序完美契合
            我们来看看代码中 `micro_batch_idx` 的生成顺序。无论是 Warmup 阶段还是 1F1B 阶段，`run_forward(micro_batch_idx)` 的调用顺序在所有 Rank 上都是严格递增的：
            * `micro_batch_idx` 的执行顺序永远是：`0, 1, 2, 3, 4, 5, 6, 7`。

            **Rank 0 的发送顺序：**
            异步发送 MB_0 -> 异步发送 MB_1 -> 异步发送 MB_2 ...

            **Rank 1 的接收顺序：**
            同步接收第 1 个到达的数据（赋给 MB_0） -> 同步接收第 2 个到达的数据（赋给 MB_1）...

            因为发送端是按 `0, 1, 2...` 的顺序塞入网络管道的，接收端也是按 `0, 1, 2...` 的顺序从管道里往外拿的。管道本身不能超车（FIFO），所以数据绝对是一一对应的，不需要任何 `tag` 来标识。

            ---

            ### 💡 代码中的一个神来之笔：`async_requests`

            在你的代码中，有一行非常关键，体现了极高的工程素养：
            ```python
            req: Optional[Work] = comms.isend_forward(output.detach())
            async_requests.append(req)  # Keep request alive to prevent buffer deallocation
            ```
            **为什么要把它 append 到一个列表里？**
            因为 `isend` 是异步的！当 `isend` 函数返回时，数据**可能还没有真正发送到网络上**，它还在当前 GPU 的显存（`output` tensor）里。
            如果在 Python 层面，这个 `output` 变量的生命周期结束了（比如被垃圾回收了，或者显存被复用了），而底层的网卡还没来得及把数据发出去，网卡就会读到**被污染的脏数据**，甚至导致段错误（Segfault）。

            把 `req` 保存到 `async_requests` 列表中，就相当于告诉 PyTorch：“这个异步任务还没结束，请你死死保住这块显存，千万别回收！”直到整个训练 step 结束，这些请求才会被释放，从而保证了异步通信的绝对安全。
            """
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