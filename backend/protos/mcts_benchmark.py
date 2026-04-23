"""
benchmark_mcts.py
-----------------
Benchmark all MCTS implementations and compare performance.
"""

import time
import chess
import torch
import multiprocessing as mp
import numpy as np
from network import ChessNet


def benchmark_serial(net, device, num_sims=800):
    """Benchmark the original serial MCTS."""
    from mcts_simple import MCTS
    
    mcts = MCTS(net, device, num_sims=num_sims, c_puct=2.5)
    board = chess.Board()
    
    start = time.time()
    root = mcts.run(board, add_noise=False)
    elapsed = time.time() - start
    
    moves, probs = mcts.get_policy(root)
    top_move = board.san(moves[probs.argmax()])
    
    return {
        'time': elapsed,
        'sims_per_sec': num_sims / elapsed,
        'top_move': top_move,
        'total_sims': num_sims,
    }


def benchmark_batched(net, device, num_sims=800, batch_size=16):
    """Benchmark batched parallel MCTS."""
    from mcts_parallel import ParallelMCTS
    
    mcts = ParallelMCTS(
        net, device,
        num_sims=num_sims,
        batch_size=batch_size,
        c_puct=2.5,
    )
    board = chess.Board()
    
    start = time.time()
    root = mcts.run(board, add_noise=False)
    elapsed = time.time() - start
    
    moves, probs = mcts.get_policy(root)
    top_move = board.san(moves[probs.argmax()])
    
    return {
        'time': elapsed,
        'sims_per_sec': num_sims / elapsed,
        'top_move': top_move,
        'total_sims': num_sims,
        'batch_size': batch_size,
    }


def benchmark_multiprocess(net, device, num_workers=4, sims_per_worker=200, batch_size=16):
    """Benchmark multiprocess MCTS."""
    from mcts_multiprocess import MultiprocessMCTS
    
    mcts = MultiprocessMCTS(
        net, device,
        num_workers=num_workers,
        sims_per_worker=sims_per_worker,
        batch_size=batch_size,
        c_puct=2.5,
    )
    board = chess.Board()
    
    start = time.time()
    move_counts = mcts.run(board, add_noise=False)
    elapsed = time.time() - start
    
    total_sims = sum(move_counts.values())
    moves, probs = mcts.get_policy(move_counts)
    top_move = board.san(moves[probs.argmax()])
    
    return {
        'time': elapsed,
        'sims_per_sec': total_sims / elapsed,
        'top_move': top_move,
        'total_sims': total_sims,
        'num_workers': num_workers,
        'batch_size': batch_size,
    }


def run_benchmarks():
    """Run all benchmarks and display results."""
    # Setup
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("MCTS PARALLELIZATION BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"CPU Cores: {mp.cpu_count()}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    net = ChessNet(num_res_blocks=4, channels=128).to(device)
    net.eval()
    
    results = []
    
    # 1. Serial baseline
    print("\n[1/5] Running serial MCTS (baseline)...")
    try:
        result = benchmark_serial(net, device, num_sims=400)
        result['method'] = 'Serial'
        result['speedup'] = 1.0
        results.append(result)
        print(f"  ✓ {result['sims_per_sec']:.1f} sims/sec")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    baseline_sps = results[0]['sims_per_sec'] if results else 1.0
    
    # 2. Batched (small batch)
    print("\n[2/5] Running batched MCTS #1...")
    try:
        result = benchmark_batched(net, device, num_sims=1600, batch_size=256)
        result['method'] = 'Batched (bs=8)'
        result['speedup'] = result['sims_per_sec'] / baseline_sps
        results.append(result)
        print(f"  ✓ {result['sims_per_sec']:.1f} sims/sec ({result['speedup']:.1f}x)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 3. Batched (larger batch)
    print("\n[3/5] Running batched MCTS #2...")
    try:
        result = benchmark_batched(net, device, num_sims=1600, batch_size=64)
        result['method'] = 'Batched (bs=16)'
        result['speedup'] = result['sims_per_sec'] / baseline_sps
        results.append(result)
        print(f"  ✓ {result['sims_per_sec']:.1f} sims/sec ({result['speedup']:.1f}x)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 4. Multiprocess (2 workers)
    print("\n[4/5] Running multiprocess MCTS (2 workers)...")
    try:
        result = benchmark_multiprocess(net, device, num_workers=4, sims_per_worker=400, batch_size=16)
        result['method'] = '2 Workers'
        result['speedup'] = result['sims_per_sec'] / baseline_sps
        results.append(result)
        print(f"  ✓ {result['sims_per_sec']:.1f} sims/sec ({result['speedup']:.1f}x)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 5. Multiprocess (4 workers)
    print("\n[5/5] Running multiprocess MCTS (4 workers)...")
    try:
        result = benchmark_multiprocess(net, device, num_workers=4, sims_per_worker=400, batch_size=32)
        result['method'] = '4 Workers'
        result['speedup'] = result['sims_per_sec'] / baseline_sps
        results.append(result)
        print(f"  ✓ {result['sims_per_sec']:.1f} sims/sec ({result['speedup']:.1f}x)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Display results table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Total Sims':<12} {'Time (s)':<10} {'Sims/sec':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['method']:<20} {r['total_sims']:<12} {r['time']:<10.2f} {r['sims_per_sec']:<12.1f} {r['speedup']:<10.1f}x")
    
    print("=" * 70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    best = max(results, key=lambda x: x['sims_per_sec'])
    print(f"  Best performer: {best['method']} ({best['sims_per_sec']:.1f} sims/sec)")
    
    if device.type == 'cuda':
        print(f"\n  For production use:")
        print(f"    - Start with: MultiprocessMCTS(num_workers=4, batch_size=16)")
        print(f"    - Expected: ~{best['sims_per_sec']:.0f} sims/sec")
        print(f"    - Monitor GPU usage with: nvidia-smi -l 1")
        print(f"    - Adjust num_workers if GPU < 80% utilized")
    else:
        print(f"\n  CPU-only mode detected:")
        print(f"    - Consider using a GPU for 10-100x speedup")
        print(f"    - CPU performance: ~{baseline_sps:.0f} sims/sec")
    
    # Move agreement check
    print("\n  Move agreement:")
    top_moves = [r['top_move'] for r in results]
    if len(set(top_moves)) == 1:
        print(f"    ✓ All methods agree on best move: {top_moves[0]}")
    else:
        print(f"    ⚠ Methods disagree (different random seeds):")
        for r in results:
            print(f"      {r['method']}: {r['top_move']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_benchmarks()