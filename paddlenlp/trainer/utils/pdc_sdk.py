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

import json
import os
import subprocess
import time
from enum import Enum
from typing import List

from paddlenlp.utils.log import logger

#
PDC_AGENT_BIN = "/root/paddlejob/tools/agent"
HASH_SUM_BIN = "/root/paddlejob/afs_tool/bin/b3sum"
TAR_BIN = "tar"


class PDCErrorCode(Enum):
    """错误码类型枚举"""

    # success
    Success = 0

    RemotePathNotExist = 1404
    LocalPathExist = 1405
    DownloadFail = 1406
    AgentConfigInvalid = 1407
    AFSToolsNotExist = 1408
    LocalPathNotExist = 1409

    CommandFail = 1501
    CalculateHashFail = 1502
    InvalidArgument = 1503


class PDCTools:
    """PDCTools"""

    def __init__(self):
        """ """
        self._pdc_agent_bin = PDC_AGENT_BIN
        self._hash_sum_bin = HASH_SUM_BIN
        self._tar_bin = TAR_BIN

    def pdc_upload(self, remote_path: str, local_path: str) -> PDCErrorCode:
        """upload data to afs/bos
        1. tar local path
        2. calculate the hash of tar file
        3. upload tar to remote path

        Args:
        remote_path str: the remote file path, afs/bos, such as afs://user/a/b/xx.tar
        local_path str: local file path

        Return:
        """
        pre_check_status = self._pre_check()
        if pre_check_status != PDCErrorCode.Success:
            return pre_check_status
        # check local path
        if not os.path.exists(local_path):
            logger.error(f"{local_path} not exist")
            return PDCErrorCode.LocalPathNotExist
        if not remote_path.endswith(".tar"):
            logger.warning(f"remote path {remote_path} should end with .tar")
            return PDCErrorCode.InvalidArgument

        try:
            # get tar name
            remote_dir = os.path.dirname(remote_path)
            tar_file_name = os.path.basename(remote_path)
            # tar local path
            status = self._tar_file(local_path, tar_file_name)
            if status != PDCErrorCode.Success:
                logger.error(f"tar local path {local_path} failed")
                return status
            # calc hash
            b3sum_hash, status = self._calculate_hash(tar_file_name)
            if status != PDCErrorCode.Success:
                logger.error(f"calculate hash for {tar_file_name} failed")
                return status
            logger.info(f"local tar: {tar_file_name}, b3sum hash: {b3sum_hash}")

            # upload local tar to remote path
            status = self._upload_file(tar_file_name, remote_path)
            if status != PDCErrorCode.Success:
                logger.error(f"upload file {tar_file_name} failed")
                return status

            # upload b3sum hash to remote path
            local_b3sum_hash_file = f".{time.time()}_b3sum.hash"
            with open(local_b3sum_hash_file, "w") as f:
                f.write(b3sum_hash)
            remote_b3sum_path = os.path.join(remote_dir, self._get_file_hash_name(tar_file_name))
            status = self._upload_file(local_b3sum_hash_file, remote_b3sum_path)
            if status != PDCErrorCode.Success:
                logger.error(f"upload hash file {local_b3sum_hash_file} failed")
                return status

            # clean tmp files
            self._clean_tmp_files([local_b3sum_hash_file, tar_file_name])

            logger.info(f"successfully uploaded ${local_path} to remote path ${remote_path}")
            return PDCErrorCode.Success
        except Exception as e:
            logger.error(f"pdc upload failed: {e}")
            raise e

    def pdc_download(self, remote_path: str, local_path: str) -> PDCErrorCode:
        """download data from afs/bos

        Args:
        remote_path str: the remote file path, afs/bos, such as afs://user/a/b/xx.tar
        local_path str: local file directory

        Return:
        """
        pre_check_status = self._pre_check()
        if pre_check_status != PDCErrorCode.Success:
            return pre_check_status
        # check local path
        if os.path.exists(local_path):
            logger.info(f"local path {local_path} already exists")
            return PDCErrorCode.LocalPathExist
        if not remote_path.endswith(".tar"):
            logger.warning(f"remote path {remote_path} should end with .tar")
            return PDCErrorCode.InvalidArgument

        try:
            remote_dir = os.path.dirname(remote_path)
            file_name = os.path.basename(remote_path)
            # download remote file to local tmp path
            local_tmp_file_path = f".tmp_{time.time()}_{file_name}"
            status = self._download_file(remote_path, local_tmp_file_path)
            if status != PDCErrorCode.Success:
                logger.error(f"download remote file {file_name} failed")
                return status

            # download hash file to local path
            hash_file_name = self._get_file_hash_name(file_name)
            hash_file_path = os.path.join(remote_dir, hash_file_name)
            status = self._download_file(hash_file_path, hash_file_name)
            if status != PDCErrorCode.Success:
                logger.error(f"download remote hash file {hash_file_path} failed")
                return status
            remote_hash = ""
            with open(hash_file_name, "r") as f:
                remote_hash = f.read().strip()

            # calc hash
            local_hash, status = self._calculate_hash(local_tmp_file_path)
            if status != PDCErrorCode.Success:
                logger.error(f"calculate hash for {local_tmp_file_path} failed")
                return status
            logger.info(f"remote hash: {remote_hash}, local hash: {local_hash}")
            # check hash
            if local_hash != remote_hash:
                logger.error(f"local b3sum hash: {local_hash}, remote b3sum hash: {remote_hash}")
                return PDCErrorCode.CalculateHashFail

            # untar file to local_path
            status = self._untar_file(local_tmp_file_path, local_path)
            if status != PDCErrorCode.Success:
                logger.error(f"untar file {local_tmp_file_path} failed")
                return status
            # clean tmp files
            self._clean_tmp_files([local_tmp_file_path])
            return PDCErrorCode.Success
        except Exception as e:
            logger.error(f"pdc upload failed: {e}")
            raise e

    def pdc_download_checkpoint(self, step: int) -> PDCErrorCode:
        """ "download checkpoint from afs/bos

        Args:
        step int: the step of checkpoint

        """
        pre_check_status = self._pre_check()
        if pre_check_status != PDCErrorCode.Success:
            return pre_check_status

        conf = json.dumps(
            {
                "download_step": step,
            }
        )
        # download file from remote path
        download_cmd_args = [
            self._pdc_agent_bin,
            "-mode",
            "command",
            "-type",
            "download_checkpoint",
            "-config",
            f"{conf}",
        ]
        try:
            self._pre_check()
            logger.info(f"begin to download checkpoint from step {step}, config: {conf}")
            res, error_code = self._exec_cmd(download_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"download checkpoint from step {step} successfully")
            return error_code
        except Exception as e:
            logger.error(f"exec cmd {download_cmd_args} with error: {e}")
            raise Exception(f"exec cmd {download_cmd_args} with error: {e}")

    def _pre_check(self) -> PDCErrorCode:
        """check whether the environment is ready"""
        if not os.path.exists(self._pdc_agent_bin):
            logger.error(f"pdc tool {self._pdc_agent_bin} not found")
            return PDCErrorCode.AFSToolsNotExist
        if not os.path.exists(self._hash_sum_bin):
            logger.error(f"hash tool {self._hash_sum_bin} not found")
            return PDCErrorCode.AFSToolsNotExist
        # TODO add more check
        return PDCErrorCode.Success

    def _exec_cmd(self, cmd_args: List[str]) -> (str, PDCErrorCode):
        """exec user command

        Args:
        cmd List[str]: command
        """
        error_code = PDCErrorCode.Success
        try:
            result = subprocess.run(cmd_args, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"exec cmd {cmd_args} failed, exit code: {result.returncode}, err: {result.stderr}")
                # TODO 区分不同的错误码:
                error_code = PDCErrorCode.CommandFail
            return result.stdout, error_code
        except Exception as e:
            logger.error(f"exec cmd {cmd_args} with error: {e}")
            raise Exception(f"exec cmd {cmd_args} with error: {e}")

    def _get_file_hash_name(self, file_name: str) -> str:
        """get the hash name of file

        Args:
        file_name str: file name

        Return:
        """
        if file_name.endswith(".tar"):
            file_name = file_name[:-4]
        return f"{file_name}.b3sumhash"

    def _calculate_hash(self, file_path: str) -> (str, PDCErrorCode):
        """calc the hash of file using b3sum

        Args:
        file_path str: file path

        Return:
        """
        cmd_args = [self._hash_sum_bin, file_path]
        try:
            result, error_code = self._exec_cmd(cmd_args)
            if error_code == PDCErrorCode.Success and len(result) > 0:
                return result.split(" ")[0].strip(), error_code
        except Exception as e:
            logger.error(f"exec cmd {cmd_args} with error: {e}")
            raise Exception(f"exec cmd {cmd_args} with error: {e}")
        return "", PDCErrorCode.CalculateHashFail

    def _tar_file(self, source_path: str, target_path: str) -> PDCErrorCode:
        """tar file with command
           tar -cf target_path -C source_path .

        Args:
        source_path str: source file path for tar
        target_path str: target file path
        """
        if not os.path.exists(source_path):
            logger.error(f"file {source_path} not exist")
            return PDCErrorCode.LocalPathNotExist
        if os.path.exists(target_path):
            os.rename(target_path, f"{target_path}.old")
            logger.warning(f"{target_path} already exists, backup it")

        error_code = PDCErrorCode.Success
        # tar file
        tar_cmd_args = [self._tar_bin, "-cf", target_path, "-C", source_path, "."]
        try:
            res, error_code = self._exec_cmd(tar_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"tar {source_path} successfully")
        except Exception as e:
            logger.error(f"exec cmd {tar_cmd_args} failed, error: {e}")
            raise Exception(f"exec cmd {tar_cmd_args} failed, error: {e}")
        return error_code

    def _untar_file(self, source_path: str, target_path: str) -> PDCErrorCode:
        """untar file
        Args:
        source_path str: source file path for untar
        target_path str: target file path
        """
        if not os.path.exists(source_path):
            logger.error(f"{source_path} not exist")
            return PDCErrorCode.LocalPathNotExist
        if not os.path.exists(target_path):
            # create target path if not exists
            os.makedirs(target_path)

        # untar file
        error_code = PDCErrorCode.Success
        untar_cmd_args = [self._tar_bin, "-xf", source_path, "-C", target_path]
        try:
            res, error_code = self._exec_cmd(untar_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"untar {source_path} successfully")
        except Exception as e:
            logger.error(f"exec cmd {untar_cmd_args} with error: {e}")
            raise Exception(f"exec cmd {untar_cmd_args} with error: {e}")
        return error_code

    def _upload_file(self, local_file_path: str, remote_path: str) -> PDCErrorCode:
        """upload file
        Args:
        local_file_path str: local file path
        remote_path str: remote file path
        """
        if not os.path.exists(local_file_path):
            logger.error(f"{local_file_path} not exist")
            return PDCErrorCode.LocalPathNotExist

        conf = json.dumps({"remote_path": remote_path, "local_path": local_file_path})
        # upload file to remote path
        upload_cmd_args = [self._pdc_agent_bin, "-mode", "command", "-type", "upload", "-config", f"{conf}"]
        error_code = PDCErrorCode.Success
        try:
            res, error_code = self._exec_cmd(upload_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"upload {local_file_path} successfully")
        except Exception as e:
            logger.error(f"exec cmd {upload_cmd_args} with error: {e}")
            raise Exception(f"exec cmd {upload_cmd_args} with error: {e}")
        return error_code

    def _download_file(self, remote_path: str, local_path: str) -> PDCErrorCode:
        """download file

        Args:
        remote_path str: remote file path
        local_path str: local file path
        """
        if os.path.exists(local_path):
            os.rename(local_path, f"{local_path}.old")
            logger.warning(f"{local_path} already exists, backup it to {local_path}.old")

        conf = json.dumps({"remote_path": remote_path, "local_path": local_path})
        # download file from remote path
        download_cmd_args = [self._pdc_agent_bin, "-mode", "command", "-type", "download", "-config", f"{conf}"]
        error_code = PDCErrorCode.Success
        try:
            logger.info(f"begin to download {remote_path}, config: {conf}")
            res, error_code = self._exec_cmd(download_cmd_args)
            if error_code == PDCErrorCode.Success:
                logger.info(f"download {remote_path} successfully")
        except Exception as e:
            logger.error(f"exec cmd {download_cmd_args} with error: {e}")
            raise Exception(f"exec cmd {download_cmd_args} with error: {e}")
        return error_code

    def _clean_tmp_files(self, tmp_files: List[str]):
        """clean tmp files

        Args:
        tmp_files List[str]: list of tmp file paths
        """
        if len(tmp_files) == 0:
            return
        # clean files
        for file_path in tmp_files:
            if os.path.exists(file_path):
                logger.info(f"clean tmp file: {file_path}")
                os.remove(file_path)
