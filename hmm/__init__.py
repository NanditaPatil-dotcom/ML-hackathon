"""HMM / Viterbi map-matching layer for PathSnap.

Public API:
    RoadGraph      - road network (nodes + directed/undirected segments)
    match_trip     - run Viterbi over one trip, return best road_id sequence
"""
from .graph import RoadGraph
from .matcher import match_trip, MatchResult

__all__ = ["RoadGraph", "match_trip", "MatchResult"]
