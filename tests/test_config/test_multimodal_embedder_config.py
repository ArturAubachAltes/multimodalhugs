import pytest

from multimodalhugs.models.multimodal_embedder.configuration_multimodal_embedder import MultiModalEmbedderConfig


def test_config_max_length_default():
    config = MultiModalEmbedderConfig()

    assert config.max_length == 200


def test_config_max_length_nondefault():
    config = MultiModalEmbedderConfig(max_length=15)

    assert config.max_length == 15


def test_config_use_backbone_max_length_true():
    config = MultiModalEmbedderConfig(pretrained_backbone="Helsinki-NLP/opus-mt-en-de", use_backbone_max_length=True)

    assert config.max_length == 512

def test_config_use_backbone_max_length_fails_without_backbone_config():
    with pytest.raises(ValueError):
        _ = MultiModalEmbedderConfig(use_backbone_max_length=True)
