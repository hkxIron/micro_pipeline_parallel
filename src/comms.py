import os
from typing import Optional

import torch
import torch.distributed as dist

"""
Rank（全局排名）
范围：0 到 (world_size - 1)

含义：进程在整个集群中的唯一标识符

用途：用于跨节点的进程间通信（MPI-like）

例子：4个节点，每个节点8个GPU，那么有32个进程，rank从0到31

Local Rank（本地排名）
范围：0 到 (每个节点的GPU数量 - 1)

含义：进程在当前节点/机器内的唯一标识符

用途：用于设置当前进程使用哪个GPU

例子：每个节点8个GPU，local_rank从0到7

总结
Rank：全局唯一标识，用于跨节点通信和全局协调
Local Rank：本地GPU标识，用于设备绑定和节点内协调

记住这个简单的规则：
需要操作硬件（GPU）时 → 用 local_rank
需要与其他进程通信时 → 用 rank

================================
错误：
# 错误：使用rank来选择GPU
device = torch.device(f"cuda:{rank}")  # 如果rank=4，但节点只有4个GPU，会出错！

# 错误：假设local_rank唯一
if local_rank == 0:  # 每个节点的local_rank=0都会执行！
    save_checkpoint()

正确：
# 正确：使用local_rank选择GPU
device = torch.device(f"cuda:{local_rank}")

# 正确：使用rank来做全局唯一操作
if rank == 0:  # 只有全局的第一个进程执行
    save_checkpoint()
"""
def init_distributed():
    """
    Initializes the distributed process group.
    Reads state directly from environment variables set by torchrun.
    """
    # 1. Read Environment Variables (set by torchrun)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"]) # NOTE: local_rank仅用于分配GPU

    # 2. Set Device
    if torch.cuda.is_available():
        # each conditional statement returns the device type
        # NOTE: 注意，device是通过local_rank来初始化的
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        # >>> torch.device("mps")
        # device(type='mps')
        # device = torch.device("mps")
        # mps doesn't work :(
        device = torch.device("cpu")
    elif torch.cpu.is_available():
        device = torch.device("cpu")
    else:
        exit()

    # 3. Initialize Group, 初始化通信组
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)
    else:
        # The code dist.init_process_group(...) is a Global State Setter.
        #  It initializes the background communication threads (C++ NCCL backend).
        # It sets up the "phone lines" so Process 0 can send data to Process 1.
        # Once called, this state persists until the program ends or you call destroy_process_group().
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size) # NOTE: CPU

    return rank, world_size, device


class PipelineComms:
    """
    多个rank之间的通信类，形成一个通信队列
    通信队列的结构是：rank0 -> rank1 -> rank2 -> ... -> rankN-1

    Args:
        rank (int): Rank of the current process. 即当前进程的rank，记住是分布式，不需要for循环
        world_size (int): Total number of processes in the distributed group.
    """
    def __init__(self, rank:int, world_size:int):
        self.rank = rank # NOTE:当前进程的rank
        self.world_size = world_size
        # Define Neighbors
        # If I am Rank 0, I have no previous neighbor (None)
        self.prev_rank = rank - 1 if rank > 0 else None
        # If I am the last Rank, I have no next neighbor (None)
        self.next_rank = rank + 1 if rank < world_size - 1 else None

    def send_forward(self, tensor):
        """Send activation to the next GPU."""
        # 将tensor从当前进程发送到下一个进程, 
        # NOTE:注意是同步发送
        # NOTE: 向后面的层发送
        # .contiguous() is required before sending
        dist.send(tensor.contiguous(), dst=self.next_rank)

    def recv_forward(self, shape, device, dtype=torch.float32):
        """Receive activation from the previous GPU."""
        # We must allocate an empty buffer to receive the data
        tensor = torch.zeros(shape, dtype=dtype, device=device) # 先在GPU上分配一个空的buffer
        # NOTE: 从前面的层接收数据
        dist.recv(tensor, src=self.prev_rank)
        return tensor

    def send_backward(self, tensor):
        """Send gradients back to the previous GPU."""
        # Blocking communication (dist.send) means
        # the program waits until the send is complete
        # before proceeding, which is simple and easier
        # to reason about. Async (isend) allows overlapping
        # computation and communication,
        # increasing efficiency and complexity.
        # NOTE:注意是同步发送
        # NOTE: 向前面的层进行反向传播
        dist.send(tensor.contiguous(), dst=self.prev_rank)

    def recv_backward(self, shape, device, dtype=torch.float32):
        """Receive gradients from the next GPU."""
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        # NOTE: 从后面的层接收梯度数据
        dist.recv(tensor, src=self.next_rank)
        return tensor

    def isend_forward(self, tensor) -> Optional[torch.distributed.Work]:
        # NOTE: 异步发送
        return dist.isend(tensor.contiguous(), dst=self.next_rank)
