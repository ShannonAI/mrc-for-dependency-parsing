# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/1/6 20:53
@desc: Topk Acc, todo make a PR to pl

"""

from typing import Any, Callable, Optional

import torch

from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.utils import _input_format_classification


class TopkAccuracy(Metric):
    r"""
    Computes `Accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_:

    .. math:: \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y_i})

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.  Works with binary, multiclass, and multilabel
    data.  Accepts logits from a model output or integer class values in
    prediction.  Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None

    Example:

        >>> from pytorch_lightning.metrics import Accuracy
        >>> target = torch.tensor([0, 1])
        >>> preds = torch.tensor([[1., 2., 0., 0.], [2., 1., 0., 0.]])
        >>> accuracy = TopkAccuracy(topk=1)
        >>> accuracy(preds, target)
        tensor(0.)
        >>> accuracy = TopkAccuracy(topk=2)
        >>> accuracy(preds, target)
        tensor(1.)
    """
    def __init__(
        self,
        topk: int = 1,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold
        self.topk = topk

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        if self.topk == 1:
            preds, target = _input_format_classification(preds, target, self.threshold)
            assert preds.shape == target.shape
            self.correct += torch.sum(preds == target)
        else:
            # [N, ..., K]
            _, topk_preds = torch.topk(preds, k=min(self.topk, preds.size(-1)), dim=-1)
            self.correct += torch.sum(topk_preds == target.unsqueeze(-1))

        self.total += target.numel()

    def compute(self):
        """
        Computes accuracy over state.
        """
        return self.correct.float() / self.total


class AllTopkAccuracy(Metric):
    """get all topk acc that less or equal to a specified number"""
    def __init__(
        self,
        topk: int = 1,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )

        self.topk = topk
        self.metrics = torch.nn.ModuleList([TopkAccuracy(k) for k in range(1, topk+1)])

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for metric in self.metrics:
            metric.update(preds, target)

    def compute(self):
        return {
            f"top{metric.topk}_acc": metric.compute() for metric in self.metrics
        }
