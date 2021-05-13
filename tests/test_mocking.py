#
#  SPDX-FileCopyrightText: 2021 Chair of Software Engineering II, University of Passau
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
from unittest import mock

from software_analysis_examples.mocking import Coverage, compute, main, uni_passau


@mock.patch("software_analysis_examples.mocking.expensive_api_call")
def test_compute_decorated(mocked_function):
    """Test the compute function using the mock as decorator

    Args:
        mocked_function: The mock of the function
    """
    expected = 124
    mocked_function.return_value = 123
    actual = compute(1)
    assert actual == expected


def test_compute_with_block():
    """Test the compute function creating the mock in a with statement."""
    expected = 124
    with mock.patch(
        "software_analysis_examples.mocking.expensive_api_call"
    ) as mocked_function:
        mocked_function.return_value = 123
        actual = compute(1)
    assert actual == expected


def test_uni_passau(mocker):
    """Test the dummy_foo function using the mocker fixture from pytest-mock.

    Args:
        mocker: The mocker fixture from pytest-mock
    """
    expected = "The City of Passau is great."
    function_mock = mocker.patch("software_analysis_examples.mocking.dummy_foo")
    function_mock.return_value = "The City of Passau"
    actual = uni_passau()
    assert expected == actual


def test_main():
    with mock.patch("software_analysis_examples.mocking._execute_tests") as m:
        m.return_value = Coverage("bar")
        main([""])
        m.assert_called_once()
