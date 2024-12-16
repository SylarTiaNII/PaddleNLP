# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# import atexit
# import copy
import multiprocessing

import paddle
from paddle.base import core
from paddle.optimizer.fusion_utils import FusionStorageHelper

from paddlenlp.utils.log import logger

# import os
# import random

# import numpy as np


DO_FUSE_OPTIMIZER = 0
DO_SYNC_PARAM = 1
DO_RETURN_DICT = 2


def get_fused_param_mappings(optimizer, manipulated_state_dict):
    param_mappings = {}
    ipc_meta_mappings = {}
    index = 0
    sharding_comm_buffers = optimizer._comm_buffer_list
    for buffer in sharding_comm_buffers:
        # Assuming all the parameters excluded from master weights are float32
        if buffer._params[0].dtype != paddle.float32:
            continue
        ipc_meta_mappings[str(index)] = buffer.param_buffer_ipc_meta
        for k, v in manipulated_state_dict.items():
            logger.info(
                f"check vname: {v.name}; buffer._sharding_param_grad_view: {buffer._sharding_param_grad_view.keys()}"
            )
            if v.name in buffer._sharding_param_grad_view:
                assert k not in param_mappings, f"{k} has already been mapped, which is unexpected."
                param_meta = {}
                param_meta["buffer_index"] = str(index)
                param_meta["shape"] = v.shape
                param_meta["name"] = v.name
                param_meta["start"] = buffer._sharding_param_grad_view[v.name]._index
                param_meta["end"] = param_meta["start"] + v._numel()
                param_mappings[k] = param_meta
        index += 1
    assert len(manipulated_state_dict) == len(
        param_mappings
    ), f"manipulated state dict is not fully covered in param mappings, manipulated_state_dict:{manipulated_state_dict.keys()}, param_mappings:{param_mappings.keys()}"
    return param_mappings, ipc_meta_mappings


class FusionWorker(multiprocessing.Process):
    def __init__(self, worker_id, device_id, task_queue, result_queue):
        super().__init__()
        self.worker_id = worker_id
        self.device_id = device_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.fusion_storage_helper = None

    def run(self):
        core.set_cuda_current_device_id(self.device_id)
        paddle.set_device(f"gpu:{self.device_id}")
        while True:
            task = self.task_queue.get()
            if task is None:
                self.task_queue.put(None)
                self.result_queue.put((self.worker_id, None))
                break

            task_type, task_body = task
            if task_type == DO_FUSE_OPTIMIZER:
                self.build_fusion_storage_helper(task_body)
            elif task_type == DO_SYNC_PARAM:
                self.fusion_storage_helper.sync_param()
                self.fusion_storage_helper.wait_all()
            elif task_type == DO_RETURN_DICT:
                result = self.fusion_storage_helper.state_dict()
                self.result_queue.put((self.worker_id, result))
            else:
                raise ValueError(f"Unknown task type: {task_type}")

    def build_fusion_storage_helper(self, task_body):
        (
            accumulators_meta,
            master_weights_meta,
            merged_model_params_meta,
            buffer_ipc_meta,
        ) = task_body
        if self.fusion_storage_helper is None:
            self.fusion_storage_helper = FusionStorageHelper(
                accumulators_meta,
                master_weights_meta,
                merged_model_params_meta,
                buffer_ipc_meta,
            )
        else:
            self.fusion_storage_helper.reset_meta(
                accumulators_meta,
                master_weights_meta,
                merged_model_params_meta,
                buffer_ipc_meta,
            )
