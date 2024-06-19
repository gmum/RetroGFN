from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Literal, Sequence

from gflownet.api.trajectories import Trajectories


@dataclass
class ArtifactOutput:
    name: str
    content: Any
    type: Literal["txt", "json", "to_pickle"]


class ArtifactsBase(ABC):
    """
    The base class for artifacts used in Trainer.
    """

    @abstractmethod
    def compute_artifacts(self, trajectories: Trajectories) -> List[ArtifactOutput]:
        ...


class ArtifactsList:
    """
    A class representing a list of artifacts used in Trainer.
    """

    def __init__(self, artifacts: Sequence[ArtifactsBase]):
        self.artifacts = artifacts

    def compute_artifacts(self, trajectories: Trajectories) -> List[ArtifactOutput]:
        artifacts_list = [artifact.compute_artifacts(trajectories) for artifact in self.artifacts]
        return [artifact for artifacts in artifacts_list for artifact in artifacts]
