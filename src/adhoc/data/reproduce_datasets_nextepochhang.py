# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch


class MyIterableDataset(torch.utils.data.IterableDataset):
# class MyIterableDataset(torch.utils.data.Dataset):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))


dataset = MyIterableDataset(0, 4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, drop_last=False)

for epoch in range(2):
    for i, data in enumerate(dataloader):
        print(i, data)

"""
stdout:
0 tensor([0, 1])
1 tensor([2, 3])
2 _IterableDatasetStopIteration(worker_id=0)
"""