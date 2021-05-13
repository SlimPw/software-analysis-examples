#
#  SPDX-FileCopyrightText: 2021 Chair of Software Engineering II, University of Passau
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
import dataclasses
import time
from typing import List

from software_analysis_examples.dummy import dummy_foo


def compute(x):
    response = expensive_api_call()
    return response + x


def expensive_api_call():
    time.sleep(1000)  # takes 1,000 seconds
    return 123


def uni_passau():
    base_string = dummy_foo()
    return f"{base_string} is great."


@dataclasses.dataclass
class Coverage:
    value: str


def decorate(func):
    def inner():
        print("Before")
        func()
        print("After")

    return inner


@decorate  # decorate(write)
def write():
    print("Write function")


if __name__ == "__main__":
    write()


def _execute_tests(module_name: str) -> Coverage:
    time.sleep(1000)  # takes 1,000 seconds
    return Coverage(module_name)


def main(argv: List[str]) -> int:
    _ = _execute_tests("foo")
    return 0
