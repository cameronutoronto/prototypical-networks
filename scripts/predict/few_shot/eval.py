import os
import json
import math
from tqdm import tqdm

import torch
import torchnet as tnt

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils

SAMPLE_MODES = ["one_shot", "random_equal", "random_unequal", "smallest_small_dists_batch",
                "largest_small_dists_batch", "smallest_sum_dists_batch",
                "largest_sum_dists_batch", "smallest_sum_dist_diffs_batch",
                "largest_sum_dist_diffs_batch", "smallest_sum_dist_diffs_pair_batch",
                "smallest_sum_dist_diffs1_rand_batch", "smallest_small_dists_unique_batch",
                "k_means_centers_batch", "smallest_sum_dist_diffs", "smallest_sum_dist_diffs_b",
                "smallest_sum_dist_diffs_c"]

SAMPLE_MODES = ["smallest_sum_dist_diffs", "smallest_sum_dist_diffs_b",
                "smallest_sum_dist_diffs_c"]

SAMPLE_MODES = ["meta_learning_nn"]

#SAMPLE_MODES = ["smallest_sum_dist_diffs1_rand_batch", "smallest_sum_dist_diffs_pair_batch"]

#"largest_sum_dist_diffs_pair_batch" is broken
def main(opt):
    # load model
    model = torch.load(opt['model.model_path'])
    model.eval()

    # load opts
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # Postprocess arguments
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')

    # construct data
    data_opt = { 'data.' + k: v for k,v in filter_opt(model_opt, 'data').items() }

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k,v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    data = data_utils.load(data_opt, ['test'])

    if data_opt['data.cuda']:
        model.cuda()
    #SAMPLE_MODES = ["smallest_sum_dist_diffs_batch"]
    meters = { field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields'] }
    for sample_mode in SAMPLE_MODES:
        model_utils.evaluate(model, data['test'], meters, sample_mode, desc="test")

        for field,meter in meters.items():
            mean, std = meter.value()
            print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(data_opt['data.test_episodes'])))
