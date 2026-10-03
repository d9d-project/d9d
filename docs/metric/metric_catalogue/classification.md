# Classification Metrics

## About

d9d provides distributed classification metrics: an approximate binary AUROC and metrics computed from a confusion matrix, such as accuracy, precision, recall and F-beta. Their state has a fixed size, so it does not grow with the number of samples.

## Binary AUROC

`BinaryAUROCMetric` approximates AUROC with histograms of predicted probabilities, as described by [Albakour et al., 2021](https://www.researchgate.net/publication/353020448_Fast_and_memory_efficient_AUC-ROC_approximation_for_Stream_Learning). More bins give a more accurate result and use more memory.

```python
from d9d.metric.impl.classification import BinaryAUROCMetric

auroc = BinaryAUROCMetric()
```

## Confusion Matrix-based Metrics

Accuracy, precision, recall and F-beta are all computed from a confusion matrix. `confusion_matrix_metric()` returns a builder that defines such a metric in three steps:

1.  **Problem type**: `binary`, `multiclass` or `multilabel`.
2.  **Statistic**: The formula to compute, e.g. `with_accuracy` or `with_f1`.
3.  **Aggregation**: How to combine the per-class values: `micro`, `macro`, `weighted` or `per_class`. `binary()` and `multiclass()` with `top_k` select `micro` themselves.

## Usage

### Binary Classification (Accuracy)

For a binary problem, you set a probability threshold, usually 0.5.

```python
from d9d.metric.impl.classification import confusion_matrix_metric

accuracy = (
    confusion_matrix_metric()
    .binary(threshold=0.5)
    .with_accuracy()
    .build()
)
```

### Multiclass Classification (Top-5 Accuracy)

Pass `top_k` to check whether the correct label is among the $K$ highest predicted scores. Each prediction is then a single hit or miss, so the problem becomes a binary one.

```python
from d9d.metric.impl.classification import confusion_matrix_metric

top5_acc = (
    confusion_matrix_metric()
    .multiclass(num_classes=1000, top_k=5)
    .with_accuracy()
    .build()
)
```

### Multiclass Classification (Per-Class Precision)

To inspect individual classes, use the `.per_class()` aggregation. It returns a separate value, such as precision, for every class.

```python
from d9d.metric.impl.classification import confusion_matrix_metric

per_class_precision = (
    confusion_matrix_metric()
    .multiclass(num_classes=10)
    .with_precision()
    .per_class()
    .build()
)
```

### Multilabel Classification (Macro F1 Score)

In a multilabel problem, several classes can be correct at the same time. Each class is compared against the probability threshold independently. The `.macro()` aggregation averages the statistic, e.g. the F1 score, over all classes with equal weight.

```python
from d9d.metric.impl.classification import confusion_matrix_metric

macro_f1 = (
    confusion_matrix_metric()
    .multilabel(num_classes=8, threshold=0.5)
    .with_f1()
    .macro()
    .build()
)
```

## API Reference

::: d9d.metric.impl.classification.BinaryAUROCMetric

::: d9d.metric.impl.classification.confusion_matrix_metric

::: d9d.metric.impl.classification.ConfusionMatrixMetricBuilder
