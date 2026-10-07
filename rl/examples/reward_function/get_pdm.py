from pathlib import Path
import os
from typing import Any, Dict
from dataclasses import asdict
import lzma
import pickle

import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataclasses import Trajectory



proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator: PDMSimulator = PDMSimulator(proposal_sampling=proposal_sampling)

ScorerConfig = PDMScorerConfig(progress_weight=5.0,
                               ttc_weight=5.0,
                               comfortable_weight=2.0,
                               driving_direction_horizon=1.0,
                               driving_direction_compliance_threshold=2.0,
                               driving_direction_violation_threshold=6.0,
                               stopped_speed_threshold=5e-03,
                               progress_distance_threshold=5.0)
# ScorerConfig = PDMScorerConfig(progress_weight=0.0,
#                                ttc_weight=5.0,
#                                comfortable_weight=2.0,
#                                driving_direction_horizon=1.0,
#                                driving_direction_compliance_threshold=2.0,
#                                driving_direction_violation_threshold=6.0,
#                                stopped_speed_threshold=5e-03,
#                                progress_distance_threshold=5.0)
scorer: PDMScorer = PDMScorer(proposal_sampling=proposal_sampling,
                              config=ScorerConfig)
_metric_cache_loader = None
       
        
def run_pdm_score(args):
    global _metric_cache_loader
    cache_path = os.environ.get("NAVSIM_METRIC_CACHE")
    if not cache_path:
        raise RuntimeError(
            "Set NAVSIM_METRIC_CACHE to the directory containing NavSim metric-cache files "
            "before using the PDM reward."
        )
    if _metric_cache_loader is None:
        _metric_cache_loader = MetricCacheLoader(Path(cache_path))

    traj = np.array(args["traj"]).reshape((8,2))
    heading = np.array(args["heading"]).reshape((8,1))
    token = args["token"]

    score_row: Dict[str, Any] = {"token": token, "valid": True}
    metric_cache_path = _metric_cache_loader.metric_cache_paths[token]
    with lzma.open(metric_cache_path, "rb") as f:
        metric_cache: MetricCache = pickle.load(f)

    trajectory = np.hstack([traj,heading])
    trajectory = Trajectory(trajectory)
    pdm_result = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
    )
    score_row.update(asdict(pdm_result))

    return score_row




# if __name__ == "__main__":

#     tokens = []
#     valid = []
#     no_at_fault_collisions = []
#     drivable_area_compliance = []
#     driving_direction_compliance = []
#     ego_progress = []
#     time_to_collision_within_bound = []
#     comfort = []
#     score = []

#     import json
#     with open(os.environ["NAVSIM_TRAJECTORY_FILE"], "r") as f:
#         data = json.load(f)

#     for token in tqdm(data):
#         traj = [row[:2] for row in data[token]]
#         heading = [row[2] for row in data[token]]

#         args = {
#                 "traj": traj, 
#                 "token": token,
#                 "heading": heading,
#             }
        
#         result = run_pdm_score(args)

#         tokens.append(result['token'])
#         valid.append(result['valid'])
#         no_at_fault_collisions.append(result['no_at_fault_collisions'])
#         drivable_area_compliance.append(result['drivable_area_compliance'])
#         driving_direction_compliance.append(result['driving_direction_compliance'])
#         ego_progress.append(result['ego_progress'])
#         time_to_collision_within_bound.append(result['time_to_collision_within_bound'])
#         comfort.append(result['comfort'])
#         score.append(result['score'])
#         # break

#     print(len(tokens))
#     print("no_at_fault_collisions: ",np.mean(no_at_fault_collisions))
#     print("drivable_area_compliance: ",np.mean(drivable_area_compliance))
#     print("driving_direction_compliance: ",np.mean(driving_direction_compliance))
#     print("ego_progress: ",np.mean(ego_progress))
#     print("time_to_collision_within_bound: ",np.mean(time_to_collision_within_bound))
#     print("comfort: ",np.mean(comfort))
#     print("score: ",np.mean(score))
