from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperRename
from d9d.module.block.head.classification import ClassificationHead
from d9d.module.block.head.embedding import EmbeddingHead
from d9d.module.block.head.language_modelling import SplitLanguageModellingHead


def _single_vocab_name(head: SplitLanguageModellingHead) -> str:
    split_names = list(head.lm_head.keys())
    if len(split_names) != 1:
        raise ValueError("HuggingFace mappers can only process a single vocab split")

    return split_names[0]


def hf_mapper_from_huggingface_lm_head(head: SplitLanguageModellingHead, prefix: str) -> ModelStateMapper:
    """Creates a state mapper translating a HuggingFace LM head weight into the d9d format.

    Args:
        head: The language modeling head instance whose vocab layout drives the rename.
        prefix: The FQN prefix under which the head lives in the composed model (e.g. ``heads.lm.``).

    Returns:
        A state mapper renaming ``lm_head.weight`` to ``{prefix}lm_head.{vocab}.weight``.
    """
    vocab_name = _single_vocab_name(head)
    return ModelStateMapperRename(name_from="lm_head.weight", name_to=f"{prefix}lm_head.{vocab_name}.weight")


def hf_mapper_to_huggingface_lm_head(head: SplitLanguageModellingHead, prefix: str) -> ModelStateMapper:
    """Creates a state mapper translating a d9d LM head weight back into the HuggingFace format.

    Args:
        head: The language modeling head instance whose vocab layout drives the rename.
        prefix: The FQN prefix under which the head lives in the composed model (e.g. ``heads.lm.``).

    Returns:
        A state mapper renaming ``{prefix}lm_head.{vocab}.weight`` to ``lm_head.weight``.
    """
    vocab_name = _single_vocab_name(head)
    return ModelStateMapperRename(name_from=f"{prefix}lm_head.{vocab_name}.weight", name_to="lm_head.weight")


def hf_mapper_from_huggingface_cls_head(head: ClassificationHead, prefix: str) -> ModelStateMapper:
    """Creates a state mapper translating a HuggingFace classification head into the d9d format.

    Args:
        head: The classification head instance.
        prefix: The FQN prefix under which the head lives in the composed model (e.g. ``heads.cls.``).

    Returns:
        A state mapper renaming ``score.weight`` to ``{prefix}score.weight``.
    """
    return ModelStateMapperRename(name_from="score.weight", name_to=f"{prefix}score.weight")


def hf_mapper_to_huggingface_cls_head(head: ClassificationHead, prefix: str) -> ModelStateMapper:
    """Creates a state mapper translating a d9d classification head back into the HuggingFace format.

    Args:
        head: The classification head instance.
        prefix: The FQN prefix under which the head lives in the composed model (e.g. ``heads.cls.``).

    Returns:
        A state mapper renaming ``{prefix}score.weight`` to ``score.weight``.
    """
    return ModelStateMapperRename(name_from=f"{prefix}score.weight", name_to="score.weight")


def hf_mapper_from_huggingface_embedding_head(head: EmbeddingHead, prefix: str) -> ModelStateMapper:
    """Creates a state mapper for a d9d embedding head against a bare HuggingFace backbone.

    The HuggingFace reference for an embedding model is the bare backbone with no head weights,
    so the embedding head has nothing to load. A trained projection cannot be sourced from
    HuggingFace and stays at its initialization.

    Args:
        head: The embedding head instance.
        prefix: The FQN prefix under which the head lives in the composed model (e.g. ``heads.embedding.``).

    Returns:
        An empty state mapper.
    """
    return ModelStateMapperParallel([])


def hf_mapper_to_huggingface_embedding_head(head: EmbeddingHead, prefix: str) -> ModelStateMapper:
    """Creates a state mapper translating a d9d embedding head back into the HuggingFace format.

    Args:
        head: The embedding head instance.
        prefix: The FQN prefix under which the head lives in the composed model (e.g. ``heads.embedding.``).

    Returns:
        An empty state mapper (the bare HuggingFace backbone has no head weights).

    Raises:
        ValueError: If the head has a trained embedding projection, which has no HuggingFace counterpart.
    """
    if head.projection is not None:
        raise ValueError("Cannot convert a model with trained embedding projection back to HuggingFace")

    return ModelStateMapperParallel([])
