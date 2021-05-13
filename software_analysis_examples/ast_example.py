#
#  SPDX-FileCopyrightText: 2021 Chair of Software Engineering II, University of Passau
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Some examples on AST handling."""
import ast
from typing import Any, Optional, Tuple

import astor  # type: ignore


def foo():
    return """def gcd(x, y):
    while y:
        x, y = y, x % y
    return x
"""


def bar():
    return """def baz(a, b, c, d, e, f, g):
    pass
"""


class NameRewriter(ast.NodeTransformer):
    """A node transformer that rewrites Name and arg nodes."""

    def visit_Name(self, node: ast.Name) -> Any:
        super().generic_visit(node)
        node.id = "foo"
        return node

    def visit_arg(self, node: ast.arg) -> Any:
        super().generic_visit(node)
        node.arg = "bar"
        return node


class ArgumentCounter(ast.NodeVisitor):
    """A node visitor that counts arguments."""

    def __init__(self) -> None:
        self._found_violation: Tuple[bool, str] = (False, "")
        self._function_name: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._function_name = node.name
        super().generic_visit(node)
        self._function_name = None

    def visit_arguments(self, node: ast.arguments) -> Any:
        super().generic_visit(node)
        if len(node.args):
            assert self._function_name is not None
            self._found_violation = (True, self._function_name)

    @property
    def found_violation(self) -> Tuple[bool, str]:
        """Provides the violation."""
        return self._found_violation


def main():
    """Execution starts here."""
    foo_code = foo()
    foo_tree = ast.parse(foo_code)
    new_tree = NameRewriter().visit(foo_tree)
    result = astor.to_source(new_tree)
    print(result)

    bar_code = bar()
    bar_tree = ast.parse(bar_code)
    counter = ArgumentCounter()
    counter.visit(bar_tree)
    print(f"Violation found? {counter.found_violation}")


if __name__ == "__main__":
    main()
