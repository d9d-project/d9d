import torch


class ExpertTokenGroups:
    """How the permuted tokens are split between consecutive experts."""

    def __init__(self, counts: torch.Tensor, host_counts: torch.Tensor | None = None):
        """Constructs the ``ExpertTokenGroups`` object.

        Args:
            counts: Tokens per expert on the device the permuted tokens live on. Shape: ``(num_experts,)``.
            host_counts: The same counts on the host, if the producer already has them. Shape: ``(num_experts,)``.
        """
        self._counts = counts
        self._host_counts = host_counts

        self._offsets = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=counts.device)
        torch.cumsum(counts, dim=0, out=self._offsets[1:])

    @property
    def offsets(self) -> torch.Tensor:
        """Token offsets of the experts with a leading zero, on the device. Shape: ``(num_experts + 1,)``."""
        return self._offsets

    def host_counts(self) -> torch.Tensor:
        """Returns the tokens per expert on the host.

        If the producer did not provide them, the first call copies them from the device and synchronizes with it.
        Later calls return the same tensor.

        Returns:
            The tokens per expert. Shape: ``(num_experts,)``.
        """
        if self._host_counts is None:
            self._host_counts = self._counts.cpu()
        return self._host_counts
