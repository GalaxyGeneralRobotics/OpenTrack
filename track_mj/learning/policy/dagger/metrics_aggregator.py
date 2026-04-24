# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logger for training metrics."""

import collections
import logging
from jax import numpy as jnp
import numpy as np
import torch
import torch.distributed as dist

class EpisodeMetricsLogger:
    """Logs training metrics for each episode."""

    def __init__(self, buffer_size=500, steps_between_logging=1e5, progress_fn=None):
        self._rollout_metrics_buffer = collections.defaultdict(lambda: collections.deque(maxlen=buffer_size))
        self._average_metrics_buffer = collections.defaultdict(lambda: collections.deque(maxlen=buffer_size))
        self._training_metrics_buffer = collections.defaultdict(lambda: collections.deque(maxlen=buffer_size))
        self._buffer_size = buffer_size
        self._steps_between_logging = steps_between_logging
        self._num_steps = 0
        self._last_log_steps = 0
        self._log_count = 0
        self._progress_fn = progress_fn

    def update_episode_metrics(self, aggregated_metrics, dones, training_metrics):
        r"""
        Args:
            aggregated_metrics: dict of jnp.ndarray
                aggregated_metrics: [#envs, ]
            dones: jnp.ndarray, [#envs, ]
            training_metrics: scaler
        """
        self._num_steps += 1
        
        if jnp.sum(dones) > 0:
            for name, metric in aggregated_metrics.items():
                done_metrics = metric[dones.astype(bool)].flatten().tolist()
                if "average_" in name:
                    self._average_metrics_buffer[name.replace("average_", "")].extend(done_metrics)
                else:
                    self._rollout_metrics_buffer[name].extend(done_metrics)
        for name, metric in training_metrics.items():
            self._training_metrics_buffer[name].append(metric)
        
        if self._num_steps - self._last_log_steps >= self._steps_between_logging:
            self.log_metrics()
            self._last_log_steps = self._num_steps

    def log_metrics(self, pad=35):
        """Log metrics to console."""
        self._log_count += 1
        log_string = f"\n{'Steps':>{pad}} Env: {self._num_steps} Log: {self._log_count}\n"
        episode_metrics = {}
        average_metrics = {}
        training_metrics = {}
        for metric_name in self._rollout_metrics_buffer:
            episode_metrics[metric_name] = np.mean(self._rollout_metrics_buffer[metric_name]).astype(np.float64)
            log_string += f"{f'Episode {metric_name}:':>{pad}}" f" {episode_metrics[metric_name]:.4f}\n"
        for metric_name in self._average_metrics_buffer:
            average_metrics[metric_name] = np.mean(self._average_metrics_buffer[metric_name]).astype(np.float64)
            log_string += f"{f'Average {metric_name}:':>{pad}}" f" {average_metrics[metric_name]:.4f}\n"
        for metric_name in self._training_metrics_buffer:
            training_metrics[metric_name] = np.mean(self._training_metrics_buffer[metric_name]).astype(np.float64)
            log_string += f"{f'Training {metric_name}:':>{pad}}" f" {training_metrics[metric_name]:.4f}\n"
        
        logging.info(log_string)

        if self._progress_fn is not None:
            episode_metrics = {f"episode/{name}": value for name, value in episode_metrics.items()}
            average_metrics = {f"average/{name}": value for name, value in average_metrics.items()}
            training_metrics = {f"training/{name}": value for name, value in training_metrics.items()}
            self._progress_fn(
                int(self._num_steps),
                {**episode_metrics, **average_metrics, **training_metrics},
            )


class DDPMetricsLogger:
    """Wrapper for EpisodeMetricsLogger that aggregates metrics across DDP ranks."""
    
    def __init__(self, base_logger: EpisodeMetricsLogger, rank, world_size, use_ddp):
        self.base_logger = base_logger
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp

        self.device_th = torch.device(f"cuda:0")        # each process only sees one GPU
    
    def update_episode_metrics(self, aggregated_metrics, dones, training_metrics):
        """Update metrics locally on each rank (no communication)."""

        # 1) Ensure rollout / average metrics keys are created on all ranks,
        #    even if there are no done steps.
        if aggregated_metrics is not None:
            for name in aggregated_metrics.keys():
                if "average_" in name:
                    # only touch, no append data
                    _ = self.base_logger._average_metrics_buffer[name.replace("average_", "")]
                else:
                    _ = self.base_logger._rollout_metrics_buffer[name]

        # 2) Ensure training metrics keys are also aligned
        if training_metrics is not None:
            for name in training_metrics.keys():
                _ = self.base_logger._training_metrics_buffer[name]

        # 3) Normal logic (append data, increase step count, etc.)
        self.base_logger.update_episode_metrics(
            aggregated_metrics, dones, training_metrics,
        )
    
    def aggregate_and_log(self):
        """Aggregate buffers across all ranks and log (only called at logging time)."""
        if not self.use_ddp:
            # Single GPU: just log normally.
            self.base_logger.log_metrics()
            return
        
        # Collect all buffer data from all ranks.
        def gather_buffer_data(buffer_dict):
            """Gather all values from a buffer dict across ranks."""
            gathered = {}

            # Unified sorted key order to avoid different rank insertion orders.
            metric_names = sorted(buffer_dict.keys())

            for name in metric_names:
                deque_obj = buffer_dict[name]
                local_data = list(deque_obj)

                # Even if local_data is empty, it should participate in all_gather.
                local_tensor = torch.tensor(
                    local_data,
                    dtype=torch.float32,
                    device=self.device_th,
                )
                local_count = torch.tensor(
                    [len(local_data)],
                    dtype=torch.int32,
                    device=self.device_th,
                )

                # 1) Gather counts.
                count_list = [
                    torch.zeros(1, dtype=torch.int32, device=self.device_th)
                    for _ in range(self.world_size)
                ]
                dist.all_gather(count_list, local_count)

                # If all ranks are 0, directly record empty results.
                max_count = max(c.item() for c in count_list)
                if max_count == 0:
                    gathered[name] = []
                    continue

                # 2) Pad to max_count.
                if local_tensor.numel() < max_count:
                    padding = torch.zeros(
                        max_count - local_tensor.numel(),
                        dtype=torch.float32,
                        device=self.device_th,
                    )
                    local_tensor = torch.cat([local_tensor, padding])

                # 3) Gather padded data.
                gather_list = [
                    torch.zeros(max_count, dtype=torch.float32, device=self.device_th)
                    for _ in range(self.world_size)
                ]
                dist.all_gather(gather_list, local_tensor)

                # 4) Remove padding, concatenate into python list.
                all_data = []
                for rank_data, rank_count in zip(gather_list, count_list):
                    valid_count = rank_count.item()
                    if valid_count > 0:
                        all_data.extend(rank_data[:valid_count].tolist())

                gathered[name] = all_data

            return gathered
        
        # Aggregate all three buffer types
        if self.rank == 0:
            global_rollout = gather_buffer_data(self.base_logger._rollout_metrics_buffer)
            global_average = gather_buffer_data(self.base_logger._average_metrics_buffer)
            global_training = gather_buffer_data(self.base_logger._training_metrics_buffer)
            
            # Temporarily replace buffers with aggregated data
            original_rollout = self.base_logger._rollout_metrics_buffer.copy()
            original_average = self.base_logger._average_metrics_buffer.copy()
            original_training = self.base_logger._training_metrics_buffer.copy()
            
            # Create new buffers with aggregated data
            for name, data in global_rollout.items():
                self.base_logger._rollout_metrics_buffer[name] = collections.deque(
                    data, maxlen=self.base_logger._buffer_size
                )
            for name, data in global_average.items():
                self.base_logger._average_metrics_buffer[name] = collections.deque(
                    data, maxlen=self.base_logger._buffer_size
                )
            for name, data in global_training.items():
                self.base_logger._training_metrics_buffer[name] = collections.deque(
                    data, maxlen=self.base_logger._buffer_size
                )
            
            # Log with aggregated data
            self.base_logger.log_metrics()
            
            # Restore original buffers
            self.base_logger._rollout_metrics_buffer = original_rollout
            self.base_logger._average_metrics_buffer = original_average
            self.base_logger._training_metrics_buffer = original_training
        else:
            # Non-rank-0 processes just participate in gathering
            gather_buffer_data(self.base_logger._rollout_metrics_buffer)
            gather_buffer_data(self.base_logger._average_metrics_buffer)
            gather_buffer_data(self.base_logger._training_metrics_buffer)
