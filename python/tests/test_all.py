import pytest
import bpe_qwen


def test_sum_as_string():
    assert bpe_qwen.sum_as_string(1, 1) == "2"
