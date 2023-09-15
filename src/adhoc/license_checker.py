# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

#!/usr/bin/env python
# Modified version of https://stackoverflow.com/a/44090218
from __future__ import print_function
from collections import defaultdict
import pkg_resources

def get_pkg_license(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')

    for line in lines:
        if line.startswith('License:'):
            return line[9:]
    return '(Licence not found)'

def print_table(table):
    column_index_to_max_width = defaultdict(int)
    for row_index, row in enumerate(table):
        for cell_index, cell in enumerate(row):
            cur_max = column_index_to_max_width[cell_index]
            cell_width = len(cell)
            if cell_width > cur_max:
                column_index_to_max_width[cell_index] = cell_width
    for row_index, row in enumerate(table):
        line = ''
        for cell_index, cell in enumerate(row):
            cell_width = column_index_to_max_width[cell_index]
            line += cell.ljust(cell_width)
            line += " - "
        line = line.rstrip(" -")
        print(line)
        if row_index == 0:
            print("-" * len(line))

def print_packages_and_licenses():
    table = []
    table.append(['Package', 'License'])
    for pkg in sorted(pkg_resources.working_set, key=lambda x: str(x).lower()):
        table.append([str(pkg), get_pkg_license(pkg)])
    print_table(table)

if __name__ == "__main__":
    print_packages_and_licenses()

