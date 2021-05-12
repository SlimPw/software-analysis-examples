#
#  SPDX-FileCopyrightText: 2021 Chair of Software Engineering II, University of Passau
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
import sys
from unittest.mock import MagicMock

import pytest
from bytecode import Bytecode, Instr, Label

from software_analysis_examples.controldependencies import CFG, ProgramGraphNode


@pytest.fixture(scope="module")
def conditional_jump_example_bytecode() -> Bytecode:
    label_else = Label()
    label_print = Label()
    byte_code = Bytecode(
        [
            Instr("LOAD_NAME", "print"),
            Instr("LOAD_NAME", "test"),
            Instr("POP_JUMP_IF_FALSE", label_else),
            Instr("LOAD_CONST", "yes"),
            Instr("JUMP_FORWARD", label_print),
            label_else,
            Instr("LOAD_CONST", "no"),
            label_print,
            Instr("CALL_FUNCTION", 1),
            Instr("LOAD_CONST", None),
            Instr("RETURN_VALUE"),
        ]
    )
    return byte_code


@pytest.fixture(scope="module")
def small_control_flow_graph() -> CFG:
    cfg = CFG(MagicMock())
    entry = ProgramGraphNode(index=0)
    n2 = ProgramGraphNode(index=2)
    n3 = ProgramGraphNode(index=3)
    n4 = ProgramGraphNode(index=4)
    n5 = ProgramGraphNode(index=5)
    n6 = ProgramGraphNode(index=6)
    exit_node = ProgramGraphNode(index=sys.maxsize)
    cfg.add_node(entry)
    cfg.add_node(n2)
    cfg.add_node(n3)
    cfg.add_node(n4)
    cfg.add_node(n5)
    cfg.add_node(n6)
    cfg.add_node(exit_node)
    cfg.add_edge(entry, n6)
    cfg.add_edge(n6, n5)
    cfg.add_edge(n5, n4)
    cfg.add_edge(n5, n3)
    cfg.add_edge(n4, n2)
    cfg.add_edge(n3, n2)
    cfg.add_edge(n2, exit_node)
    return cfg


@pytest.fixture(scope="module")
def larger_control_flow_graph() -> CFG:
    graph = CFG(MagicMock())
    entry = ProgramGraphNode(index=-sys.maxsize)
    n_1 = ProgramGraphNode(index=1)
    n_2 = ProgramGraphNode(index=2)
    n_3 = ProgramGraphNode(index=3)
    n_5 = ProgramGraphNode(index=5)
    n_100 = ProgramGraphNode(index=100)
    n_110 = ProgramGraphNode(index=110)
    n_120 = ProgramGraphNode(index=120)
    n_130 = ProgramGraphNode(index=130)
    n_140 = ProgramGraphNode(index=140)
    n_150 = ProgramGraphNode(index=150)
    n_160 = ProgramGraphNode(index=160)
    n_170 = ProgramGraphNode(index=170)
    n_180 = ProgramGraphNode(index=180)
    n_190 = ProgramGraphNode(index=190)
    n_200 = ProgramGraphNode(index=200)
    n_210 = ProgramGraphNode(index=210)
    n_300 = ProgramGraphNode(index=300)
    n_exit = ProgramGraphNode(index=sys.maxsize)
    graph.add_node(entry)
    graph.add_node(n_1)
    graph.add_node(n_2)
    graph.add_node(n_3)
    graph.add_node(n_5)
    graph.add_node(n_100)
    graph.add_node(n_110)
    graph.add_node(n_120)
    graph.add_node(n_130)
    graph.add_node(n_140)
    graph.add_node(n_150)
    graph.add_node(n_160)
    graph.add_node(n_170)
    graph.add_node(n_180)
    graph.add_node(n_190)
    graph.add_node(n_200)
    graph.add_node(n_210)
    graph.add_node(n_300)
    graph.add_node(n_exit)
    graph.add_edge(entry, n_1)
    graph.add_edge(n_1, n_2)
    graph.add_edge(n_2, n_3)
    graph.add_edge(n_3, n_5)
    graph.add_edge(n_5, n_100)
    graph.add_edge(n_100, n_110)
    graph.add_edge(n_110, n_120, label="true")
    graph.add_edge(n_120, n_130)
    graph.add_edge(n_130, n_140)
    graph.add_edge(n_140, n_150, label="true")
    graph.add_edge(n_150, n_160)
    graph.add_edge(n_160, n_170, label="false")
    graph.add_edge(n_170, n_180)
    graph.add_edge(n_180, n_190)
    graph.add_edge(n_160, n_190, label="true")
    graph.add_edge(n_190, n_140)
    graph.add_edge(n_140, n_200, label="false")
    graph.add_edge(n_200, n_210)
    graph.add_edge(n_210, n_110)
    graph.add_edge(n_110, n_300, label="false")
    graph.add_edge(n_300, n_exit)
    return graph
