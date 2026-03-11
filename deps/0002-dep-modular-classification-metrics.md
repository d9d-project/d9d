---
DEP: 0002
Title: Modular Classification Metrics Architecture
Author: Daniil Sergeev @DaniilSergeev17
Status: Implemented
Type: Feature
Created: 2026-02-13
---

# DEP-0002: Modular Classification Metrics Architecture

## Abstract

This proposal introduces a modular architecture for classification metrics in `d9d` based on composable pieces.

The design preserves `Metric` lifecycle compatibility (`update/sync/compute/reset/to/state_dict/load_state_dict`) and supports binary, multiclass, and multilabel classification using a fluent Builder API without deep inheritance trees.

## Motivation

Current and expected classification metrics often become hard to maintain due to:

1. Coupled logic: state updates and aggregation policy are implemented together.
2. Inconsistent state forms: every metric stores different tensors and reductions.
3. Repetition: micro/macro/weighted logic is duplicated per metric family.
4. Feature pressure: top-k, thresholded, and multilabel variants become separate classes even when they share the same statistic.

We want a framework-level metric composition model that:

- Preserves `d9d` distributed metric lifecycle.
- Produces deterministic and testable behavior.
- Supports extensibility using a fluent builder interface rather than inheritance-heavy APIs.

## Design Proposal

### 1. Core Data Structure: `ConfusionMatrix`

The fundamental unit of state is the **Confusion Matrix**. We treat classification tasks as a collection of binary problems (One-vs-Rest). Thus, a `ConfusionMatrix` represents the $2 \times 2$ components (TP, FP, TN, FN) for one or multiple classes.

```python
import dataclasses
import torch

@dataclasses.dataclass(kw_only=True, slots=True)
class ConfusionMatrix:
    """Represents a confusion matrix for classification evaluation."""
    tp: torch.Tensor
    fp: torch.Tensor
    tn: torch.Tensor
    fn: torch.Tensor
```

When accumulating metrics, these tensors have the shape `(C,)` where `C` is the number of classes. For simple binary evaluation, `C = 1`.

### 2. Statistics (`ConfusionMatrixStatistic`)

Statistics are **stateless** protocols operating on a `ConfusionMatrix`. They calculate a raw score (like F1 or Precision) for the classes based strictly on the current counts.

```python
from typing import Protocol
import torch

class ConfusionMatrixStatistic(Protocol):
    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        ...

class PrecisionStatistic(ConfusionMatrixStatistic):
    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        return matrix.tp / (matrix.tp + matrix.fp)
```

### 3. Aggregators (`ConfusionMatrixAggregator`)

Aggregators define how per-class tensors returned by `ConfusionMatrixStatistic` are combined into a final tensor. They **own the statistic**, ensuring aggregation logic is decoupled from pure mathematical formulation.

```python
from enum import StrEnum
import torch

class ClassificationAggregationMethod(StrEnum):
    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = "none"

class ConfusionMatrixAggregator:
    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        ...
```

### 4. Accumulator (`ConfusionMatrixAccumulator`)

We need only **one** accumulator. The `ConfusionMatrixAccumulator` maintains four 1D `MetricAccumulators` (TP, FP, TN, FN) corresponding to the number of output channels, wrapping safe distributed behavior for these primitives.

```python
class ConfusionMatrixAccumulator(Stateful):
    def __init__(self, num_outputs: int):
        self._num_outputs = num_outputs
        self._tp = MetricAccumulator(torch.zeros(num_outputs, dtype=torch.long))
        # ... internal tn/fp/fn definitions omitted for brevity

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Flattened batch dimensions:
        preds = preds.long().flatten(0, -2)
        targets = targets.long().flatten(0, -2)

        # Vectorized calculation over the batch
        tp = (preds * targets).sum(dim=0)
        fp = (preds * (1 - targets)).sum(dim=0)
        fn = ((1 - preds) * targets).sum(dim=0)
        tn = ((1 - preds) * (1 - targets)).sum(dim=0)

        self._tp.update(tp)
        # ... logic
```

### 5. Processors (`ClassificationPredictionsProcessor`)

Processors normalize raw model outputs into canonical binary tensor pairs `(preds, targets)` aligned strictly against matrices required by the accumulator.

```python
class ClassificationPredictionsProcessor(Protocol):
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

class TopKProcessor(ClassificationPredictionsProcessor):
    ...

class OneHotProcessor(ClassificationPredictionsProcessor):
    ...

class ThresholdProcessor(ClassificationPredictionsProcessor):
    ...
```

### 6. Composed Metric Type

`ConfusionMatrixMetric` composes processor, accumulator, and aggregator into a single `Metric`-compatible pipeline.

### 7. Builder API

To guarantee logical composition without overwhelming kwargs or manual assembly, the pipeline is constructed securely using the `ConfusionMatrixMetricBuilder`.

The builder requires specific sequential choices:
1. **Problem Type**: `.binary()`, `.multiclass()`, `.multilabel()` — Defines the `Processor` and accumulator counts.
2. **Statistic**: `.with_accuracy()`, `.with_f1()`, `.with_precision()`, etc. — Defines the base mathematical protocol.
3. **Aggregation Context**: `.micro()`, `.macro()`, `.weighted()`, `.per_class()` — Defines the reduction strategy.

## Usage

### Using The Builder Fluent API (Recommended)

**1. Binary Precision**
```python
from d9d.metric.impl.classification import confusion_matrix_metric

metric = (
    confusion_matrix_metric()
    .binary(threshold=0.5)
    .with_precision()
    .build()
)
```

**2. Binary F1 with custom threshold**
```python
from d9d.metric.impl.classification import confusion_matrix_metric

metric = (
    confusion_matrix_metric()
    .binary(threshold=0.3)
    .with_f1()
    .build()
)
```

**3. Multiclass Precision (Macro, 10 classes)**
```python
from d9d.metric.impl.classification import confusion_matrix_metric

metric = (
    confusion_matrix_metric()
    .multiclass(num_classes=10)
    .with_precision()
    .macro()
    .build()
)
```

**4. Multiclass F1 (Weighted, 5 classes)**
```python
from d9d.metric.impl.classification import confusion_matrix_metric

metric = (
    confusion_matrix_metric()
    .multiclass(num_classes=5)
    .with_f1()
    .weighted()
    .build()
)
```

**5. Top-5 Accuracy**
```python
from d9d.metric.impl.classification import confusion_matrix_metric

metric = (
    confusion_matrix_metric()
    .multiclass(num_classes=1000, top_k=5)
    .with_accuracy()  # note: top_k defaults to MICRO aggregation automatically
    .build()
)
```

**6. Multilabel Per-Class FBeta**
```python
from d9d.metric.impl.classification import confusion_matrix_metric

metric = (
    confusion_matrix_metric()
    .multilabel(num_classes=8, threshold=0.5)
    .with_fbeta(beta=2.0)
    .per_class()
    .build()
)
```

### Manual Component Assembly

For deeply custom environments, `ConfusionMatrixMetric` can be assembled manually:

**1. Top-5 Recall Pipeline Explicitly**

```python
from d9d.metric.component.classification import (
    ClassificationAggregationMethod,
    ConfusionMatrixAccumulator,
    ConfusionMatrixAggregator,
    RecallStatistic,
    TopKProcessor,
)
from d9d.metric.impl.classification import ConfusionMatrixMetric

metric = ConfusionMatrixMetric(
    processor=TopKProcessor(k=5),
    accumulator=ConfusionMatrixAccumulator(num_outputs=1),
    aggregator=ConfusionMatrixAggregator(
        method=ClassificationAggregationMethod.MICRO,
        statistic=RecallStatistic(),  # Recall = TP / (TP + FN) = Hits / Total
    ),
)
```

## Backward Compatibility
Will this break existing code using d9d? Will old training scripts fail?

**Yes**. This PR introduces breaking changes to the `d9d.metric.impl` module structure:
1. `BinaryAccuracyMetric` has been entirely removed in favor of the builder pattern.
2. The flat import structure in `d9d.metric.impl` has been heavily refactored into sub-namespaces (`aggregation`, `classification`, `container`). Code importing strings like `from d9d.metric.impl import ComposeMetric, SumMetric` will break.

**Migration Plan:**
Because `d9d` is still evolving core frameworks, we accept structural breaks for long-term maintainability. Users must update their imports and instantiation calls:
*   Change `BinaryAccuracyMetric(threshold=0.5)` to `confusion_matrix_metric().binary(threshold=0.5).with_accuracy().build()`.
*   Change `from d9d.metric.impl import ComposeMetric` to `from d9d.metric.impl.container import ComposeMetric`.
*   Change `from d9d.metric.impl import SumMetric, WeightedMeanMetric` to `from d9d.metric.impl.aggregation import SumMetric, WeightedMeanMetric`.

## Alternatives Considered
Why is this design better than other options you thought of?

**1. Deep Inheritance Hierarchy (TorchMetrics approach)**
*   *Idea*: Create an abstract `Metric` base and subclass it into `BinaryAccuracy`, `MulticlassAccuracy`, `MultilabelAccuracy`, `BinaryF1`, etc.
*   *Drawbacks*: This leads to a massive class explosion, combinatorial scaling (Task Type $\times$ Statistic $\times$ Aggregation Method). It duplicates distributed accumulator code across various endpoints and makes adding a new concept (like a custom formula) require a massive inheritance footprint.
*   *Why ours is better*: Composition completely eliminates inheritance explosion.

**2. Monolithic Configurable Metric**
*   *Idea*: A single `ClassificationMetric(task="multiclass", num_classes=10, statistic="f1", average="macro")` handler.
*   *Drawbacks*: The constructor becomes an unmaintainable chain of `if/else` statements. Mutually exclusive arguments (e.g., passing `threshold` for multiclass mode) are either ignored silently or bloat the initialization with runtime validation checks. It is also closed to custom statistics or processors without modifying d9d core.
*   *Why ours is better*: The `ConfusionMatrixMetricBuilder` provides sequential interface safety (you literally cannot call `.macro()` until you've configured a domain that supports it), and the protocol-based backend allows developers to drop in their own `ConfusionMatrixStatistic` without forking the repository. 

**3. Direct Factory Functions**
*   *Idea*: Write specific factory functions like `f1_score(num_classes, average)`.
*   *Drawbacks*: While we initially proposed standard factory functions, we realized it still limits the user to predefined formulas. The builder pattern offers the exact same boilerplate reduction as factory functions but preserves complete extensibility and autocomplete IDE support through a fluent API chain.
