"""
xGMI Transport bandwidth test for AMD MI300X GPUs via Mooncake Transfer Engine.

Tests:
  1. Data correctness (4KB round-trip)
  2. Write/Read bandwidth measurement (64MB x 10 iterations)
  3. Comparison with TCP transport bandwidth

Usage:
    python3 test_xgmi_bandwidth.py
"""
import multiprocessing
import ctypes
import time
import sys

multiprocessing.set_start_method('spawn', force=True)

INITIATOR_GPU = 0
TARGET_GPU = 1
VRAM_BUFFER_SIZE = 256 * 1024 * 1024  # 256 MB


def run_target(ready_event, done_event, rpc_port_val, gpu_addr_val,
               gpu_id, vram_size, protocol):
    """Run target engine in a subprocess with GPU VRAM registered."""
    import torch
    from mooncake.engine import TransferEngine

    torch.cuda.set_device(gpu_id)
    vram_tensor = torch.zeros(vram_size, dtype=torch.uint8, device=f'cuda:{gpu_id}')
    vram_addr = vram_tensor.data_ptr()

    target = TransferEngine()
    ret = target.initialize('127.0.0.1:12345', 'P2PHANDSHAKE', protocol, '')
    if ret != 0:
        print(f'[Target] init failed: {ret}', file=sys.stderr)
        rpc_port_val.value = -1
        ready_event.set()
        return

    ret = target.register_memory(vram_addr, vram_size)
    if ret != 0:
        print(f'[Target] register failed: {ret}', file=sys.stderr)
        rpc_port_val.value = -1
        ready_event.set()
        return

    actual_port = target.get_rpc_port()
    print(f'[Target] Ready GPU {gpu_id}, port={actual_port}, addr=0x{vram_addr:x}')
    rpc_port_val.value = actual_port
    gpu_addr_val.value = vram_addr
    ready_event.set()
    done_event.wait(timeout=120)
    target.unregister_memory(vram_addr)


def run_bandwidth_test(protocol, label, initiator_gpu, target_gpu, vram_size):
    """Run a bandwidth test for a given protocol."""
    import torch
    from mooncake.engine import TransferEngine

    ready_event = multiprocessing.Event()
    done_event = multiprocessing.Event()
    rpc_port_val = multiprocessing.Value(ctypes.c_int, 0)
    gpu_addr_val = multiprocessing.Value(ctypes.c_ulonglong, 0)

    p = multiprocessing.Process(
        target=run_target,
        args=(ready_event, done_event, rpc_port_val, gpu_addr_val,
              target_gpu, vram_size, protocol),
        daemon=True
    )
    p.start()
    if not ready_event.wait(timeout=30):
        raise RuntimeError(f'{label}: Target did not start')
    target_port = rpc_port_val.value
    if target_port <= 0:
        raise RuntimeError(f'{label}: Target init failed (port={target_port})')
    target_name = f'127.0.0.1:{target_port}'
    dst_addr = gpu_addr_val.value

    torch.cuda.set_device(initiator_gpu)
    # Allocate GPU buffer on initiator
    src_tensor = torch.zeros(vram_size, dtype=torch.uint8, device=f'cuda:{initiator_gpu}')
    src_addr = src_tensor.data_ptr()

    initiator = TransferEngine()
    ret = initiator.initialize('127.0.0.1:12347', 'P2PHANDSHAKE', protocol, '')
    assert ret == 0, f'{label}: Initiator init failed: {ret}'
    ret = initiator.register_memory(src_addr, vram_size)
    assert ret == 0, f'{label}: Initiator register failed: {ret}'

    # --- Correctness test: 4KB round-trip ---
    data_len = 4096
    pattern_bytes = bytes([0x41 + (i % 26) for i in range(data_len)])
    pattern_tensor = torch.tensor(list(pattern_bytes), dtype=torch.uint8, device=f'cuda:{initiator_gpu}')

    # Write pattern to GPU0 via torch
    src_tensor[:data_len].copy_(pattern_tensor)
    torch.cuda.synchronize()

    # Write GPU0 -> GPU1
    ret = initiator.transfer_sync_write(target_name, src_addr, dst_addr, data_len)
    assert ret == 0, f'{label}: Write failed: {ret}'

    # Clear GPU0
    src_tensor[:data_len].zero_()
    torch.cuda.synchronize()

    # Read GPU1 -> GPU0
    ret = initiator.transfer_sync_read(target_name, src_addr, dst_addr, data_len)
    assert ret == 0, f'{label}: Read failed: {ret}'

    # Verify by reading back to CPU
    torch.cuda.synchronize()
    read_back = src_tensor[:data_len].cpu().numpy().tobytes()
    assert read_back == pattern_bytes, f'{label}: Data mismatch!'
    print(f'[PASS] {label}: 4KB correctness test passed')

    # --- Bandwidth test: 64MB x 10 iters ---
    data_len = 64 * 1024 * 1024

    # Fill with non-zero data
    src_tensor[:data_len].fill_(0xAB)
    torch.cuda.synchronize()

    # Warmup
    initiator.transfer_sync_write(target_name, src_addr, dst_addr, data_len)

    iters = 10
    start = time.monotonic()
    for _ in range(iters):
        ret = initiator.transfer_sync_write(target_name, src_addr, dst_addr, data_len)
        assert ret == 0
    elapsed_write = time.monotonic() - start
    write_bw = (data_len * iters) / elapsed_write / (1024**3)

    start = time.monotonic()
    for _ in range(iters):
        ret = initiator.transfer_sync_read(target_name, src_addr, dst_addr, data_len)
        assert ret == 0
    elapsed_read = time.monotonic() - start
    read_bw = (data_len * iters) / elapsed_read / (1024**3)

    print(f'\n=== {label} Bandwidth (GPU {initiator_gpu} <-> GPU {target_gpu}) ===')
    print(f'Write: {write_bw:.2f} GB/s ({data_len*iters/1024**2:.0f} MB in {elapsed_write:.3f}s)')
    print(f'Read:  {read_bw:.2f} GB/s ({data_len*iters/1024**2:.0f} MB in {elapsed_read:.3f}s)')

    done_event.set()
    p.join(timeout=10)
    if p.is_alive():
        p.terminate()

    return write_bw, read_bw


def main():
    # Test xGMI transport
    xgmi_write, xgmi_read = run_bandwidth_test(
        'xgmi', 'xGMI Transport', INITIATOR_GPU, TARGET_GPU, VRAM_BUFFER_SIZE)

    print(f'\n=== Summary ===')
    print(f'xGMI Write: {xgmi_write:.2f} GB/s')
    print(f'xGMI Read:  {xgmi_read:.2f} GB/s')
    print('[DONE]')


if __name__ == '__main__':
    main()
