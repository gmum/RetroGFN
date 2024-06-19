from typing import Any, Dict, List, Sequence, Set

import gin

from gflownet.api.trajectories import Trajectories
from gflownet.trainer.metrics.metric_base import MetricsBase
from gflownet.utils.helpers import ContentHeap


@gin.configurable()
class StandardGFNMetrics(MetricsBase):
    """
    A standard set of metrics for GFN. It includes the mean of the reward and the proxy.
    """

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        reward_outputs = trajectories.get_reward_outputs()
        output_dict = {
            "reward_mean": reward_outputs.reward.mean().item(),
            "proxy_mean": reward_outputs.proxy.mean().item(),
            "max_reward": reward_outputs.reward.max().item(),
            "min_reward": reward_outputs.reward.min().item(),
            "max_proxy": reward_outputs.proxy.max().item(),
            "min_proxy": reward_outputs.proxy.min().item(),
        }
        if reward_outputs.proxy_components is not None:
            proxy_components_dict = {
                f"proxy_{key}_mean": value.mean().item()
                for key, value in reward_outputs.proxy_components.items()
            }
            output_dict = output_dict | proxy_components_dict
        return output_dict


@gin.configurable()
class TopKProxyMetric(MetricsBase):
    """
    A metric that computes the mean of the top k proxy values.
    """

    def __init__(
        self, k_list: Sequence[int] = (1, 100, 1000), include_all_components: bool = False
    ):
        """

        Args:
            k_list: a list of k values to compute the top k proxy values.
            include_all_components: whether to include all components of the proxy in the output.
        """
        super().__init__()
        self.component_to_heaps: Dict[str, Dict[int, ContentHeap]] = {}
        self.k_list = k_list
        self.include_all_components = include_all_components

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        proxy_outputs = trajectories.get_reward_outputs()
        terminal_states = trajectories.get_last_states_flat()
        proxy_dict = {"proxy": proxy_outputs.proxy}
        if proxy_outputs.proxy_components is not None and self.include_all_components:
            proxy_dict.update(proxy_outputs.proxy_components)

        output_dict = {}
        for name, values in proxy_dict.items():
            if name not in self.component_to_heaps:
                self.component_to_heaps[name] = {k: ContentHeap(max_size=k) for k in self.k_list}
            for k, heap in self.component_to_heaps[name].items():
                for state, value in zip(terminal_states, values):
                    heap.push(value=value.item(), item=state)
                output_dict.update({f"top_{k}_{name}_mean": sum(el.value for el in heap) / k})

        return output_dict


@gin.configurable()
class NumModesFound(MetricsBase):
    """
    A metric that computes the number of distinct modes (states) found by the agent with proxy values above a threshold.
    """

    def __init__(self, proxy_value_threshold_list: List[float], proxy_higher_better: bool = True):
        """

        Args:
            proxy_value_threshold_list: a list of proxy value thresholds.
            proxy_higher_better: whether higher proxy values are better.
        """
        super().__init__()
        self.proxy_value_threshold_list = proxy_value_threshold_list
        self.proxy_higher_better = proxy_higher_better
        self.threshold_to_set: Dict[float, Set[Any]] = {
            threshold: set() for threshold in proxy_value_threshold_list
        }

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        reward_outputs = trajectories.get_reward_outputs()
        terminal_states = trajectories.get_last_states_flat()
        for state, proxy_value in zip(terminal_states, reward_outputs.proxy):
            for threshold in self.proxy_value_threshold_list:
                if (self.proxy_higher_better and proxy_value.item() >= threshold) or (
                    not self.proxy_higher_better and proxy_value.item() <= threshold
                ):
                    self.threshold_to_set[threshold].add(state)

        return {
            f"num_modes_{threshold}": len(self.threshold_to_set[threshold])
            for threshold in self.proxy_value_threshold_list
        }
