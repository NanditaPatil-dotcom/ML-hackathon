"""Request / response models (the API contract shared with the evaluator UI)."""
from typing import Dict, List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    trip_id: Union[str, int] = Field(validation_alias=AliasChoices("trip_id", "tripId", "id"))


class TripInfo(BaseModel):
    trip_id: str
    num_observations: int


class TripsResponse(BaseModel):
    count: int
    trips: List[TripInfo]


class RouteMetrics(BaseModel):
    observation_accuracy: float  # share of matched GPS points whose arc is on the reference route
    arc_precision: float
    arc_recall: float
    arc_f1: float
    sequence_similarity: float  # 2*LCS / (len(pred)+len(ref)) over collapsed arc sequences


class PredictResponse(BaseModel):
    trip_id: str
    num_observations: int
    num_observations_used: int
    num_outliers_removed: int
    predicted_arc_sequence: List[int]  # HMM route, consecutive duplicates collapsed
    per_observation_arcs: List[Optional[int]]  # one arc_id per GPS row (null = dropped as outlier)
    predicted_route_coords: List[List[float]]  # [[lon, lat], ...] polyline of the predicted route
    gps_points: List[List[float]]  # [[lon, lat], ...] raw observations (for the map)
    hmm_accuracy: Optional[float]  # headline number = hmm.observation_accuracy
    classifier_only_accuracy: Optional[float]
    metrics: Optional[Dict[str, RouteMetrics]]  # {"hmm": ..., "classifier_only": ...}; null if no reference
    processing_time_ms: float
    pipeline: Dict[str, str]
