# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import logging

data_pipelines = {
    'contriever256': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'd_del_ratio': 0.1,
        'pseudo_query_ratio': 1.0,
        'aug_special_query': False,
    },
}


def load_dataprocess_config(training_args, local_rank):
    logger = logging.getLogger(__name__)
    # prepare for data loader
    if training_args.data_pipeline_name:
        data_prep_config = data_pipelines[training_args.data_pipeline_name]
        if training_args.pseudo_query_names:
            pseudo_query_names = eval(training_args.pseudo_query_names)
            data_prep_config['pseudo_query_names'] = pseudo_query_names
        if local_rank == 0 or local_rank == -1:
            logger.info('Using pre-defined data pipeline: ' + str(training_args.data_pipeline_name))
    else:
        data_prep_config = {
            'max_context_len': training_args.max_context_len,
            'min_dq_len': training_args.min_dq_len,
            'min_q_len': training_args.min_q_len,
            'max_q_len': training_args.max_q_len,
            'min_d_len': training_args.min_d_len,
            'max_d_len': training_args.max_d_len,
            'word_del_ratio': training_args.word_del_ratio,
            'dq_prompt_ratio': training_args.dq_prompt_ratio,
            'pseudo_query_ratio': training_args.pseudo_query_ratio,
        }
    if local_rank == 0 or local_rank == -1:
        logger.info('Data loading parameters:')
        for k, v in data_prep_config.items():
            setattr(training_args, k, v)
            logger.info(f'\t\t{k} = {v}')

    return data_prep_config

