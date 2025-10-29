"""
GPU Validator Module

REQ-006: Ensure GPU-only operation with failure if no GPU is available.
REQ-010: All code components directly linked to requirements.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class GPUValidator:
    """
    GPU Validator to ensure GPU-only operation.

    REQ-006: Validates GPU availability and ensures no CPU fallback.
    """

    def __init__(self) -> None:
        """
        Initialize GPU validator.

        Raises:
            RuntimeError: If no GPU is available.
        """
        if not torch.cuda.is_available():
            error_msg = "REQ-006: GPU is not available. CPU fallback is disabled."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.device: torch.device = torch.device("cuda")
        gpu_name: str = torch.cuda.get_device_name(0)
        gpu_count: int = torch.cuda.device_count()

        logger.info(f"REQ-006: GPU validation successful. Found {gpu_count} GPU(s)")
        logger.info(f"REQ-006: Primary GPU: {gpu_name}")

        # Verify GPU can perform operations
        self._verify_gpu_operation()

    def _verify_gpu_operation(self) -> None:
        """
        Verify GPU can perform operations.

        REQ-006: Verify GPU compute capability.
        """
        try:
            test_tensor: torch.Tensor = torch.randn(10, 10, device=self.device)
            _result = torch.matmul(test_tensor, test_tensor)
            logger.debug("REQ-006: GPU compute verification successful")
        except Exception as e:
            error_msg = f"REQ-006: GPU compute verification failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_device(self) -> torch.device:
        """
        Get the validated GPU device.

        Returns:
            torch.device: The CUDA device.
        """
        return self.device

    def get_device_count(self) -> int:
        """
        Get the number of available GPUs.

        Returns:
            int: Number of CUDA devices.
        """
        return torch.cuda.device_count()

    def set_device(self, device_id: int) -> None:
        """
        Set the active GPU device.

        Args:
            device_id: The GPU device ID.

        Raises:
            RuntimeError: If device_id is invalid.
        """
        if device_id < 0 or device_id >= torch.cuda.device_count():
            error_msg = f"REQ-006: Invalid device_id: {device_id}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        torch.cuda.set_device(device_id)
        self.device = torch.device(f"cuda:{device_id}")
        logger.info(f"REQ-006: Set active GPU to device {device_id}")


def get_gpu_validator() -> GPUValidator | None:
    """
    Get or create GPU validator instance.

    REQ-006: Factory function for GPU validation.

    Returns:
        Optional[GPUValidator]: GPU validator if GPU is available, None otherwise.
    """
    try:
        return GPUValidator()
    except RuntimeError as e:
        logger.error(f"REQ-006: GPU validation failed: {e}")
        raise
