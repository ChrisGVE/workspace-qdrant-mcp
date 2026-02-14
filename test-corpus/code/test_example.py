"""Test file for is_test detection.

Expected: file_type=code, language=python, is_test=true
The test_ prefix in the filename should trigger is_test detection.
"""

import pytest


class TestCalculator:
    """Test suite for Calculator class."""

    def test_addition(self):
        assert 1 + 1 == 2

    def test_subtraction(self):
        assert 5 - 3 == 2

    def test_multiplication(self):
        assert 3 * 4 == 12

    def test_division(self):
        assert 10 / 2 == 5.0

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            1 / 0


@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]


def test_list_sum(sample_data):
    assert sum(sample_data) == 15


def test_list_length(sample_data):
    assert len(sample_data) == 5
