# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from collections import Counter

import datasets

stackex_dataset = datasets.load_dataset("HuggingFaceGECLM/StackExchange_Mar2023", cache_dir='/export/home/data/pretrain/.cache', split='salesforce', streaming=False)

subset_count = Counter()
length_count = Counter()

for i, ex in enumerate(stackex_dataset):
    if i % 100 == 0:
        print(i)
    if i >= 10000:
        break
    print(ex)
