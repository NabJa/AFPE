import torch
import torchmetrics as tm
from torchmetrics import MetricCollection


def get_classification_metrics(
    name: str, task="binary", num_labels=None, num_classes=None
) -> MetricCollection:

    return MetricCollection(
        {
            f"{name}/auroc": tm.AUROC(
                task=task, num_labels=num_labels, num_classes=num_classes
            ),
            f"{name}/accuracy": tm.Accuracy(
                task=task, num_labels=num_labels, num_classes=num_classes
            ),
            f"{name}/f1": tm.F1Score(
                task=task, num_labels=num_labels, num_classes=num_classes
            ),
            f"{name}/precision": tm.Precision(
                task=task, num_labels=num_labels, num_classes=num_classes
            ),
            f"{name}/recall": tm.Recall(
                task=task, num_labels=num_labels, num_classes=num_classes
            ),
        }
    )


def get_regression_metrics(name: str, num_dims=1) -> MetricCollection:
    return MetricCollection(
        {
            f"{name}/mae": tm.MeanAbsoluteError(),
            f"{name}/mse": tm.MeanSquaredError(),
            f"{name}/r2": tm.R2Score(num_outputs=num_dims),
        }
    )


def top_cls_threshold(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Threshold the prediction so that the classes with highest probability are selected."""
    number_of_classes = gt.sum(dim=1).int()
    thresholded = torch.zeros_like(pred)

    for i in range(pred.shape[0]):
        top_indices = pred[i].topk(number_of_classes[i].item()).indices
        thresholded[i, top_indices] = 1

    return thresholded


def get_multi_confusion_matrix(pred, gt):
    tp = ((gt == 1) & (pred == 1)).sum(dim=0)
    tn = ((gt == 0) & (pred == 0)).sum(dim=0)
    fp = ((gt == 0) & (pred == 1)).sum(dim=0)
    fn = ((gt == 1) & (pred == 0)).sum(dim=0)
    return tp, tn, fp, fn


def accuracy(pred, gt):
    tp, tn, fp, fn = get_multi_confusion_matrix(pred, gt)
    return (tp + tn) / (tp + tn + fp + fn)


def precision(pred, gt):
    tp, tn, fp, fn = get_multi_confusion_matrix(pred, gt)
    return tp / (tp + fp + 1e-9)


def recall(pred, gt):
    tp, tn, fp, fn = get_multi_confusion_matrix(pred, gt)
    return tp / (tp + fn + 1e-9)


class MultiLabelMetric:
    def __init__(
        self, func, num_labels, class_names=None, threshold_preds=True, name_prefix=""
    ):
        self.func = func
        self.num_labels = num_labels
        self.class_names = self._parse_class_names(class_names)
        self.threshold_preds = threshold_preds
        self.name_prefix = name_prefix
        self.reset()

    def reset(self):
        self._sum = torch.zeros(self.num_labels)
        self._count = 0

    def _parse_class_names(self, names):
        if names is None:
            return list(range(self.num_labels))
        assert len(names) == self.num_labels
        return names

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        assert pred.shape == gt.shape
        assert pred.shape[1] == self.num_labels

        pred = pred.cpu()
        gt = gt.cpu()

        if self.threshold_preds:
            # pred = threshold_prediction(pred)
            pred = top_cls_threshold(pred, gt)

        _v = self.func(pred, gt)
        self._sum += _v
        self._count += 1

        return _v

    def compute(self, reset=True):
        if self._count == 0:
            raise ValueError("Nothing to accumulate.")

        # Compute means and zip with class names
        names_and_results = zip(self.class_names, self._sum / self._count)

        # Generate result dictionary
        result = {
            f"{self.name_prefix}{name}": res.item() for name, res in names_and_results
        }

        if reset:
            self.reset()

        return result

    def mean_aggregate(self, reset=True):
        mean = torch.mean(self._sum / self._count)
        if reset:
            self.reset()
        return {
            f"{self.name_prefix}mean": mean.item(),
        }


class MultiLabelPerformance:
    def __init__(self, class_names, name_prefix="") -> None:

        n = len(class_names)

        self.performance = {
            "accuracy": MultiLabelMetric(
                accuracy,
                n,
                class_names=class_names,
                name_prefix=f"{name_prefix}accuracy_",
            ),
            "precision": MultiLabelMetric(
                precision,
                n,
                class_names=class_names,
                name_prefix=f"{name_prefix}precision_",
            ),
            "recall": MultiLabelMetric(
                recall, n, class_names=class_names, name_prefix=f"{name_prefix}recall_"
            ),
        }

    def __call__(self, pred, gt) -> None:
        for metric in self.performance.values():
            metric(pred, gt)

    def compute(self) -> dict:
        result = {}
        for metric in self.performance.values():
            result.update(metric.compute(reset=False))
            result.update(metric.mean_aggregate(reset=True))

        return result
