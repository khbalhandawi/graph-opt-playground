"""Solve the E2000_V20_T6 toy benchmark with the weighted evaluators.

  * ExactWeightedEvaluator  -> BruteForce oracle (enumerates all 203 partitions)
  * SampleWeightedEvaluator -> greedy heuristic and exact MIQP, 10 sampled realizations each

Reads ``data_toy/test_airline_nodes_E2000_V20_T6.graphml`` (see generate_toy_graph.py)
and writes to ``models_opt/E2000_V20_T6/low_samples_weighted``, which toy_problems.ipynb
loads. Run from the repository root.
"""
import networkx as nx
import os
from typing import List
import numpy as np

from graph_opt.community_detection import (
    GraphLoader, NeighbourSampler,
    ExactWeightedEvaluator, SampleWeightedEvaluator,
    GreedyOptimizer, BruteForce, ExactOptimizer,
)

if __name__ == "__main__":

    data_dir = 'data_toy'
    model_dir = 'models_opt/E2000_V20_T6/low_samples_weighted'
    USE_GPU = False
    SEED = 1234
    WALK_LENGTH = 2
    THREADS_PER_BLOCK_2D = (22, 22)
    THREADS_PER_BLOCK_3D = (8, 8, 8)
    N_SAMPLES_EDGE = 50
    N_WALKS = 10
    N_PROCESSES = 2
    WEIGHTS = (0.7, 0.3)
    RTOL = 1e-6
    EPSILON = 0.1
    N_SAMPLE_PAIRS = None
    MULTIPROCESS = False   # weighted score path is light; avoid mp semaphore churn
    N_SAMPLES = 10
    N_COMMUNITIES = None

    os.makedirs(model_dir, exist_ok=True)

    MG = nx.readwrite.graphml.read_graphml(
        os.path.join(data_dir, 'test_airline_nodes_E2000_V20_T6.graphml'))
    graph = GraphLoader(MG=MG, use_gpu=USE_GPU)

    # ---- WEIGHTED exact evaluator for the brute-force oracle ----
    exact_evaluator = ExactWeightedEvaluator(
        gloader=graph,
        use_gpu=USE_GPU,
        walk_length=WALK_LENGTH,
        threads_per_block_2D=THREADS_PER_BLOCK_2D,
        threads_per_block_3D=THREADS_PER_BLOCK_3D,
        stream=None,
    )

    ########################################
    ## 1. Optimization oracle (brute force)
    ########################################

    brute_optimizer = BruteForce(
        evaluator=exact_evaluator,
        seed=SEED,
        num_streams=N_PROCESSES,
        weights=WEIGHTS,
        use_gpu=USE_GPU,
    )
    print('[brute] enumerating partitions ...', flush=True)
    brute_optimizer.optimize(max_iterations=None, rtol=RTOL, multiprocess=MULTIPROCESS)
    brute_optimizer.save(os.path.join(model_dir, "brute_force"))
    print(f'[brute] oracle f={brute_optimizer.f_history[-1]:.6f} '
          f'(f1,f2)={brute_optimizer.multi_history[-1]}', flush=True)

    ########################################
    ## 2. Stochastic optimization (greedy)
    ########################################

    greedy_opts: List[GreedyOptimizer] = []
    for i_sample in range(N_SAMPLES):
        sampler_i = NeighbourSampler(
            gloader=graph, seed=i_sample, use_gpu=USE_GPU,
            n_samples_edge=N_SAMPLES_EDGE, n_walks=N_WALKS, walk_length=WALK_LENGTH,
        )
        sampler_i.execute_randomwalk()

        sample_evaluator_i = SampleWeightedEvaluator(
            gloader=graph, sampler=sampler_i, use_gpu=USE_GPU,
            n_communities=N_COMMUNITIES,
            threads_per_block_2D=THREADS_PER_BLOCK_2D,
            threads_per_block_3D=THREADS_PER_BLOCK_3D, stream=None,
        )

        greedy_optimizer = GreedyOptimizer(
            evaluator=sample_evaluator_i, seed=i_sample,
            num_streams=N_PROCESSES, weights=WEIGHTS, use_gpu=USE_GPU,
        )
        greedy_optimizer.optimize(
            max_iterations=10000, epsilon=EPSILON, rtol=RTOL,
            n_sample_pairs=N_SAMPLE_PAIRS, multiprocess=MULTIPROCESS,
        )
        greedy_optimizer.save(os.path.join(model_dir, f"greedy_optimizer_{i_sample}"))
        greedy_opts.append(greedy_optimizer)
        print(f'[greedy {i_sample}] f={greedy_optimizer.f_history[-1]:.6f}', flush=True)

    ########################################
    ## 3. Exact optimization (MIQP)
    ########################################

    model_settings = {
        'TimeLimit': 72_000, 'NodefileStart': 0.5, 'SoftMemLimit': 50,
        'MIPGap': 1e-3, 'Heuristics': 0.15, 'MIPFocus': 1, 'Cuts': 0,
        'Presolve': 0, 'Threads': 4, 'ScaleFlag': 1, 'Method': 1,
        'FeasibilityTol': 1e-9, 'IntFeasTol': 1e-9,
    }

    exact_opts: List[ExactOptimizer] = []
    for i_sample in range(N_SAMPLES):
        sampler_i = NeighbourSampler(
            gloader=graph, seed=i_sample, use_gpu=USE_GPU,
            n_samples_edge=N_SAMPLES_EDGE, n_walks=N_WALKS, walk_length=WALK_LENGTH,
        )
        sampler_i.execute_randomwalk()

        sample_evaluator_i = SampleWeightedEvaluator(
            gloader=graph, sampler=sampler_i, use_gpu=USE_GPU,
            n_communities=N_COMMUNITIES,
            threads_per_block_2D=THREADS_PER_BLOCK_2D,
            threads_per_block_3D=THREADS_PER_BLOCK_3D, stream=None,
        )

        model_path = os.path.join(model_dir, f"exact_optimizer_{i_sample}")
        exact_optimizer = ExactOptimizer(
            evaluator=sample_evaluator_i, seed=i_sample, weights=WEIGHTS,
            log_dir=model_path, log_interval=1, model_settings=model_settings,
        )
        exact_optimizer.build_model()
        exact_optimizer.optimize()
        exact_optimizer.save(model_path)
        exact_opts.append(exact_optimizer)
        print(f'[exact {i_sample}] f={exact_optimizer.f_history[-1]:.6f}', flush=True)

    print('\n[done] weighted toy re-run complete ->', model_dir, flush=True)
