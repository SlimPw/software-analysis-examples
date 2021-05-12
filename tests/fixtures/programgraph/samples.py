#
#  SPDX-FileCopyrightText: 2021 Chair of Software Engineering II, University of Passau
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#


def for_loop(sign, some_list):
    for index, mapping in enumerate(some_list):
        if sign in mapping:
            return index
