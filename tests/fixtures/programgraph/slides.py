#
#  SPDX-FileCopyrightText: 2021 Chair of Software Engineering II, University of Passau
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
import math


def basic():
    print("3D plot")
    print("Creative Computing")
    print()
    print()
    print()
    fna = lambda z: 30 * math.exp(-z * z / 100)  # noqa
    print()
    for x in range(-30, 30):
        l = 0  # noqa
        y_1 = 5 * int(math.sqrt(900 - x * x) / 5)
        for y in range(y_1, -y_1, -5):
            z = int(25 + fna(math.sqrt(x * x + y * y)) - 0.7 * y)
            if z <= l:
                return 190
            l = z  # noqa
            print("*")
        print()
