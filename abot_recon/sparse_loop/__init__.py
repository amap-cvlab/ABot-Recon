"""Self-contained sparse loop-closure components."""

from .graph import LoopEdge, PoseGraphConfig, build_odometry_edges, optimize_pose_graph
from .retrieval import LoopCandidate, RetrievalConfig, retrieve_candidates

__all__ = [
    "LoopCandidate",
    "LoopEdge",
    "PoseGraphConfig",
    "RetrievalConfig",
    "build_odometry_edges",
    "optimize_pose_graph",
    "retrieve_candidates",
]
