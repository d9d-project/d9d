# Classification Metrics

The `d9d` framework provides robust, distributed-ready classification metrics designed to handle large-scale data smoothly.

## Binary AUROC

```python
from d9d.metric.impl.classification import BinaryAUROCMetric

auroc = BinaryAUROCMetric()
```

::: d9d.metric.impl.classification.BinaryAUROCMetric
    options:
      heading_level: 3

## Confusion Matrix-based Metrics

When evaluating categorical outcomes, many standard statistics (Accuracy, Precision, Recall, F-Beta) share an underlying reliance on the Confusion Matrix.

To provide maximum flexibility and code reuse, `d9d` exposes a fluent, safe builder pattern via `confusion_matrix_metric()`. You define your metric in three distinct steps:
1. **Problem Type**: Define if this is `binary`, `multiclass`, or `multilabel`.
2. **Statistic**: Choose the formula to evaluate (e.g., `with_accuracy`, `with_f1`).
3. **Aggregation**: Choose how to reduce multi-dimensional data (`micro`, `macro`, `weighted`, or `per_class`).

Below are several common examples of how to assemble these configurations.

### Binary Classification (Accuracy)

For a simple binary problem, you specify a probability threshold (usually 0.5). Using `.with_accuracy()` makes the metric calculate the overall correct predictions without needing complex aggregation.

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

You can easily evaluate if the correct label appears within the top $K$ predicted probabilities by passing `top_k` into the multiclass configuration. Since `top_k` treats the evaluation as a single broad "hit or miss", it effectively becomes a binary classification problem.

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

Instead of collapsing results into a single global number, you might want to inspect the performance of strictly individual categories. Using the `.per_class()` aggregation bypasses global reductions entirely and returns a separate score (such as Precision) for every single class.

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

### Multilabel Classification (Macro F1-Score)

For multilabel problems, multiple correct categories can exist simultaneously. Each class is evaluated independently against a probability threshold. To compute a single global metric value, you can use reductions like `.macro()` to average the specific statistic (e.g., F1-score) evenly across all classes, regardless of their individual sample frequency.

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

::: d9d.metric.impl.classification.confusion_matrix_metric
    options:
      heading_level: 3

::: d9d.metric.impl.classification.ConfusionMatrixMetricBuilder
    options:
      heading_level: 3
