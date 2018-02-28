from tqdm import tqdm
import numpy as np
import torch

from protonets.utils import filter_opt
from protonets.models import get_model
from protonets.models.utils import euclidean_dist
import copy

import numpy as np
import tensorflow as tf
import random

#SAMPLE_MODE = "one_shot"
#SAMPLE_MODE = "random_equal"
#SAMPLE_MODE = "random_unequal"
#SAMPLE_MODE = "smallest_sum_dists_batch"
#SAMPLE_MODE = "largest_sum_dists_batch"

MAX_NUM_QUERY_POINTS = 14


def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)

def std_dev(items):
    mean = float(sum(items)) / len(items)
    dev = sum([items[x] - mean for x in range(len(items))])
    dev /= len(items) - 1
    return dev ** 0.5


## Check best over multiple runs 
##def evaluate(model, data_loader, meters, sample_mode="random_equal", desc=None):
##    model.eval()
##    print("Sample Mode:", sample_mode)
##    for field,meter in meters.items():
##        meter.reset()
##
##    if desc is not None:
##        data_loader = tqdm(data_loader, desc=desc)
##    num_inv_samples = 0
##
##    best_accs = []
##    best_losses = []
##    for sample in data_loader:
##        best_acc = 1e10
##        best_loss = 0
##        for x in range(10):
##            sample['valid'] = True
##            new_sample = create_new_sample(copy.deepcopy(sample), sample_mode)
##            if not new_sample['valid']:
##                #print("Invalid Sample")
##                num_inv_samples += 1
##                continue
##            _, output = model.loss(new_sample)
##            acc = output['acc']
##            loss = output['loss']
##            if (acc < best_acc):
##                best_acc = acc
##            if (loss > best_loss):
##                best_loss = loss
##            for field, meter in meters.items():
##                meter.add(output[field])
##        best_accs.append(best_acc)
##        best_losses.append(best_loss)
##    print("Avg Best Acc: ", float(sum(best_accs)) / len(best_accs), "+/-", std_dev(best_accs))
##    print("Avg Best Loss: ", float(sum(best_losses)) / len(best_losses), "+/-", std_dev(best_losses))
##    print("Invalid Samples:", num_inv_samples)
##    return meters


## Standard check best
def evaluate(model, data_loader, meters, sample_mode="random_equal", desc=None):
    model.eval()
    print("Sample Mode:", sample_mode)
    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)
    num_inv_samples = 0

    for sample in data_loader:
        sample['valid'] = True
        new_sample = create_new_sample(copy.deepcopy(sample), sample_mode, model)
        if not new_sample['valid']:
            num_inv_samples += 1
            continue
        _, output = model.loss(new_sample)
        
        for field, meter in meters.items():
            meter.add(output[field])

    print("Invalid Samples:", num_inv_samples)
    return meters

# Add Single
##def evaluate(model, data_loader, meters, sample_mode="random_equal", desc=None):
##    model.eval()
##    print("Sample Mode:", sample_mode)
##    for field,meter in meters.items():
##        meter.reset()
##
##    if desc is not None:
##        data_loader = tqdm(data_loader, desc=desc)
##    num_inv_samples = 0
##
##    for sample in data_loader:
##        sample['valid'] = True
##        new_sample,_,_ = create_new_sample(copy.deepcopy(sample), random.randint(0, 95))
##        if not new_sample['valid']:
##            #print("Invalid Sample")
##            num_inv_samples += 1
##            continue
##        _, output = model.loss(new_sample)
##        
##        for field, meter in meters.items():
##            meter.add(output[field])
##
##    print("Invalid Samples:", num_inv_samples)
##    return meters

##Get Change in Loss information
#0.105104 +- 0.024076     0.978543 +- 0.002945
##def evaluate(model, data_loader, meters, sample_mode="random_equal", desc=None):
##    print("Calculating Loss Information - Eval Results Invalid!")
##    model.eval()
##    print("Sample Mode:", sample_mode)
##    for field,meter in meters.items():
##        meter.reset()
##
##    if desc is not None:
##        data_loader = tqdm(data_loader, desc=desc)
##    num_inv_samples = 0
##
##    for sample in data_loader:
##        new_sample = create_new_sample(copy.deepcopy(sample), "one_shot")
##        _, output = model.loss(new_sample)
##        base_loss = output['loss']
##        sample['valid'] = True
##        for x in range(95):
##            new_sample, protos, test_point, counts = create_new_sample(copy.deepcopy(sample), str(x))
##            if not new_sample['valid']:
##                num_inv_samples += 1
##                continue
##            _, output = model.loss(new_sample)
##            loss = output['loss']
##            change_in_loss = base_loss - loss
##            save_change_loss_info(protos, counts, test_point, base_loss, change_in_loss)
##            for field, meter in meters.items():
##                meter.add(output[field])
##
##    print("Invalid Samples:", num_inv_samples)
##    return meters

def compress_protos(protos, counts):
    compressed = []
    for x in range(len(protos)):
        compressed.append(counts[x])
        for y in range(len(protos[x])):
            compressed.append(protos[x][y])
    return compressed

def save_change_loss_info(protos, counts, test_point, base_loss, change_in_loss):
    with open("change_in_losses.txt", "a") as f:
        compressed = compress_protos(protos, counts)
        f.write(str(counts[0]))
        for x in range(1, len(compressed)):
            f.write(","+str(compressed[x]))
        for x in range(len(test_point)):
            f.write("," + str(test_point[x]))
        f.write(",", str(base_loss))
        f.write("," + str(change_in_loss) + "\n")
        
def get_lcm(items):
    lcm = len(items[0])
    for x in range(1, len(items)):
        if lcm % len(items[x]) != 0:
            lcm *= len(items[x])
    return lcm

def get_sample_size(sample):
    n_c = len(sample['class']) #number of classes
    n_s = len(sample['xs'][0]) #number of support points per class
    n_q = len(sample['xq'][0]) #number of query points per class
    n_t = n_c * (n_s + n_q) #total number points in episode
    return n_c, n_s, n_q, n_t

def create_one_shot_sample(sample, model):
    ''' One support image per class '''
    #for x in range(5):
    #    for y in range(5):
    #        sample['xs'][x][y] = sample['xs'][x][0]
    #return sample
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds = [x * n_s for x in range(n_c)] # add one of each class
    query_inds = []
    for x in range(n_t):
        if x not in support_inds:
            query_inds.append(x)
    new_sample = get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    return new_sample

def copy_images(sample, n_c, n_s, n_q):
    images = []
    for x in range(n_c):
        for y in range(n_s):
            images.append(sample['xs'][x][y])
    for x in range(n_c):
        for y in range(n_q):
            images.append(sample['xq'][x][y])
    return images


def get_embedded_space_images(sample, model):
    ''' Return embedded points '''
    return model.embedded_points(sample)

def get_new_sample(sample, image_list, support_inds, query_inds, n_c, n_s, n_q):
    #print(len(image_list), len(support_inds), len(query_inds))
    support_points = [[] for x in range(n_c)]
    query_points = [[] for x in range(n_c)]
    diff = n_c * n_s

    for ind in support_inds:
        if (ind < diff):
            class_ind = ind // n_s
        else:
            class_ind = (ind - diff) // n_q
        #print(class_ind, ind, len(image_list))
        support_points[class_ind].append(image_list[ind])

    #print(sample['xs'].shape)

    for ind in query_inds:
        if (ind < diff):
            class_ind = ind // n_s
        else:
            class_ind = (ind - diff) // n_q

        query_points[class_ind].append(image_list[ind])

    support_lcm = get_lcm(support_points)
    query_min = MAX_NUM_QUERY_POINTS

    for x in range(len(query_points)):
        query_min = min(query_min, len(query_points[x]))

    if query_min != MAX_NUM_QUERY_POINTS:
        sample['valid'] = False
        #print("Query_Min:", query_min)
        return sample
        
    for x in range(len(support_points)):
        orig_len = len(support_points[x])
        mult_fac = int(support_lcm / orig_len)
        for y in range(mult_fac-1):
            for z in range(orig_len):
                support_points[x].append(support_points[x][y])

    for x in range(len(query_points)):
        query_points[x] = query_points[x][:query_min]
            
        
    for x in range(n_c):
        support_points[x] = torch.stack(support_points[x])
        query_points[x] = torch.stack(query_points[x])
    #print(support_points[0].shape)
    sample['xs'] = torch.stack(support_points)
    sample['xq'] = torch.stack(query_points)
    #print(sample['xs'].shape)
    #print(sample['xq'].shape)
    return sample

def create_random_equal_sample(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds = [x for x in range(n_c*n_s)] # add one of each class
    query_inds = [x for x in range(n_c*n_s, n_t)]
    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_random_unequal_sample(sample, model):
    ''' random number of support images per class (>=1)
        sum of total support images remains constant '''
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    rand_perm = torch.randperm(n_t)
    support_inds = [rand_perm[x] for x in range(n_c*n_s)]
    query_inds = [rand_perm[x] for x in range(n_c*n_s, n_t)]

    swap_loc = 0 #ensure all base support points are in support_inds
    for ind in support_inds_base:
        if ind not in support_inds:         
            q_ind = query_inds.index(ind)
            for x in range(swap_loc, len(support_inds)):
                if support_inds[swap_loc] not in support_inds_base:
                    break
                swap_loc += 1

            support_inds[swap_loc], query_inds[q_ind] = query_inds[q_ind], support_inds[swap_loc]

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_smallest_small_dists_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    dists = [0.0 for x in range(n_t)]
    for x in range(n_t):
        class_dists = [torch.dist(emb_images[x], emb_images[support_inds_base[y]]) for y in range(n_c)]
        dists[x] = min(class_dists)

    inds = [x for x in range(n_t)]
    dists, inds = zip(*sorted(zip(dists, inds)))

    num_added = n_c
    support_inds = []
    query_inds = []
    for x in range(n_t):
        ind = inds[x]
        if ind in support_inds_base:
            support_inds.append(ind)
        elif num_added < n_c*n_s:
            num_added += 1
            support_inds.append(ind)
        else:
            query_inds.append(ind)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_largest_small_dists_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    dists = [0.0 for x in range(n_t)]
    for x in range(n_t):
        class_dists = [torch.dist(emb_images[x], emb_images[support_inds_base[y]]) for y in range(n_c)]
        dists[x] = min(class_dists)

    inds = [x for x in range(n_t)]
    dists, inds = zip(*sorted(zip(dists, inds)))

    num_added = n_c
    support_inds = []
    query_inds = []
    for x in range(n_t):
        ind = inds[-1-x]
        if ind in support_inds_base:
            support_inds.append(ind)
        elif num_added < n_c*n_s:
            num_added += 1
            support_inds.append(ind)
        else:
            query_inds.append(ind)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_smallest_sum_dists_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    dists = [0.0 for x in range(n_t)]
    for x in range(n_t):
        for y in range(n_c): #use euclidean distance???
            dists[x] += torch.dist(emb_images[x], emb_images[support_inds_base[y]])

    inds = [x for x in range(n_t)]
    dists, inds = zip(*sorted(zip(dists, inds)))

    num_added = n_c
    support_inds = []
    query_inds = []
    for x in range(n_t):
        ind = inds[x]
        if ind in support_inds_base:
            support_inds.append(ind)
        elif num_added < n_c*n_s:
            num_added += 1
            support_inds.append(ind)
        else:
            query_inds.append(ind)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_largest_sum_dists_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    dists = [0.0 for x in range(n_t)]
    for x in range(n_t):
        for y in range(n_c): #use euclidean distance???
            dists[x] += torch.dist(emb_images[x], emb_images[support_inds_base[y]])

    inds = [x for x in range(n_t)]
    dists, inds = zip(*sorted(zip(dists, inds)))

    num_added = n_c
    support_inds = []
    query_inds = []
    for x in range(n_t):
        ind = inds[-1-x]
        if ind in support_inds_base:
            support_inds.append(ind)
        elif num_added < n_c*n_s:
            num_added += 1
            support_inds.append(ind)
        else:
            query_inds.append(ind)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_smallest_sum_dist_diffs_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    diffs = [0.0 for x in range(n_t)]
    for x in range(n_t):
        dists = [torch.dist(emb_images[x], emb_images[support_inds_base[y]]) for y in range(n_c)]
        min_dist = min(dists)
        min_ind = dists.index(min_dist)
        for y in range(n_c):
            dists[y] = dists[y] - min_dist
        dists[min_ind] = max(dists) #don't count it to itself
        min_diff = min(dists)
        diffs[x] = min_diff        

    inds = [x for x in range(n_t)]
    diffs, inds = zip(*sorted(zip(diffs, inds)))

    num_added = n_c
    support_inds = []
    query_inds = []
    for x in range(n_t):
        ind = inds[x]
        if ind in support_inds_base:
            support_inds.append(ind)
        elif num_added < n_c*n_s:
            num_added += 1
            support_inds.append(ind)
        else:
            query_inds.append(ind)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_largest_sum_dist_diffs_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    diffs = [0.0 for x in range(n_t)]
    for x in range(n_t):
        dists = [torch.dist(emb_images[x], emb_images[support_inds_base[y]]) for y in range(n_c)]
        min_dist = min(dists)
        min_ind = dists.index(min_dist)
        for y in range(n_c):
            dists[y] = dists[y] - min_dist
        dists[min_ind] = max(dists) #don't count it to itself
        min_diff = min(dists)
        diffs[x] = min_diff 
                         
    inds = [x for x in range(n_t)]
    diffs, inds = zip(*sorted(zip(diffs, inds)))

    num_added = n_c
    support_inds = []
    query_inds = []
    for x in range(n_t):
        ind = inds[-1-x]
        if ind in support_inds_base:
            support_inds.append(ind)
        elif num_added < n_c*n_s:
            num_added += 1
            support_inds.append(ind)
        else:
            query_inds.append(ind)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_smallest_sum_dist_diffs1_rand_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    diffs = [0.0 for x in range(n_t)]
    for x in range(n_t):
        dists = [torch.dist(emb_images[x], emb_images[support_inds_base[y]]) for y in range(n_c)]
        min_dist = min(dists)
        min_ind = dists.index(min_dist)
        for y in range(n_c):
            dists[y] = dists[y] - min_dist
        dists[min_ind] = max(dists) #don't count it to itself
        min_diff = min(dists)
        diffs[x] = min_diff        

    inds = [x for x in range(n_t)]
    diffs, inds = zip(*sorted(zip(diffs, inds)))

    num_added = n_c
    support_inds = []
    query_inds = []
    for x in range(n_t):
        ind = inds[x]
        if ind in support_inds_base:
            support_inds.append(ind)
        elif num_added < n_c+1: #add most uncertain
            num_added += 1
            support_inds.append(ind)
        else:
            query_inds.append(ind)

    for x in range(n_c+1, n_c*n_s): #add random points
        support_inds.append(query_inds[x])

    query_inds = query_inds[(n_c*n_s-(n_c+1)):]

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_smallest_sum_dist_diffs_pair_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    diffs = [[1e30 for y in range(n_c)] for x in range(n_c)]
    diffs_inds = [[-1 for y in range(n_c)] for x in range(n_c)]
    for x in range(n_t):
        dists = [torch.dist(emb_images[x], emb_images[support_inds_base[y]]) for y in range(n_c)]
        min_dist = min(dists)
        min_ind = dists.index(min_dist)
        for y in range(n_c):
            dists[y] = dists[y] - min_dist
        dists[min_ind] = 1e31 #don't count it to itself
        for y in range(n_c):
            if (dists[y] < diffs[min_ind][y]):
                diffs[min_ind][y] = dists[y]
                diffs[y][min_ind] = dists[y]
                diffs_inds[min_ind][y] = x
                diffs_inds[y][min_ind] = x

    support_inds = [ind for ind in support_inds_base]
    for x in range(n_c*n_s-n_c):
        for attempts in range(100):
            min_y = 0
            min_z = 0
            min_val = 1e30
            for y in range(n_c):
                for z in range(n_c):
                    if diffs[y][z] < min_val:
                        min_y = y
                        min_z = z
                        min_val = diffs[y][z]
                        
            ind = diffs_inds[min_y][min_z]
            diffs[min_y][min_z] = 1e30
            diffs[min_z][min_y] = 1e30
            if ind not in support_inds:
                support_inds.append(ind)
                break

    for x in range(len(support_inds), n_s*n_c):
        ind = 0
        while ind in support_inds:
            ind += 1
        support_inds.append(ind)

    query_inds = []
    for x in range(n_t):
        if x not in support_inds:
            query_inds.append(x)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_largest_sum_dist_diffs_pair_batch(sample, model):
    ''' Broken '''
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    diffs = [[0.0 for y in range(n_c)] for x in range(n_c)]
    diffs_inds = [[0 for y in range(n_c)] for x in range(n_c)]
    for x in range(n_t):
        dists = [torch.dist(emb_images[x], emb_images[support_inds_base[y]]) for y in range(n_c)]
        min_dist = min(dists)
        min_ind = dists.index(min_dist)
        for y in range(n_c):
            dists[y] = dists[y] - min_dist
        dists[min_ind] = -1 #don't count it to itself
        for y in range(n_c):
            if (dists[y] > diffs[min_ind][y]):
                diffs[min_ind][y] = dists[y]
                diffs[y][min_ind] = dists[y]
                diffs_inds[min_ind][y] = x
                diffs_inds[y][min_ind] = x

    support_inds = [ind for ind in support_inds_base]
    for x in range(n_c*n_s-n_c):
        print(diffs)
        print(diffs_inds)
        while True:
            max_y = 0
            max_z = 0
            max_val = 0.0
            for y in range(n_c):
                for z in range(n_c):
                    if diffs[y][z] > max_val:
                        max_y = y
                        max_z = z
                        max_val = diffs[y][z]
                        
            ind = diffs_inds[max_y][max_z]
            diffs[max_y][max_z] = 0.0
            diffs[max_z][max_y] = 0.0
            if ind not in support_inds:
                support_inds.append(ind)
                break

    query_inds = []
    for x in range(n_t):
        if x not in support_inds:
            query_inds.append(x)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_smallest_sum_dist_diffs(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    support_inds = [ind for ind in support_inds_base]
    for num in range(n_s*n_c - n_c):
        add_diff = 1e30
        add_ind = 0
        for x in range(n_t):
            dists = [torch.dist(emb_images[x], emb_images[support_inds[y]]) for y in range(len(support_inds))]
            min_dist = min(dists)
            min_ind = dists.index(min_dist)
            for y in range(len(support_inds)):
                dists[y] = dists[y] - min_dist
            dists[min_ind] = max(dists) #don't count it to itself
            min_diff = min(dists)
            if min_diff < add_diff:
                add_diff = min_diff
                add_ind = x

        support_inds.append(add_ind)
        

    query_inds = []
    for x in range(n_t):
        if x not in support_inds:
            query_inds.append(x)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def get_prototypes(emb_images, classes, n_c):
    prototypes = []
    
    for x in range(n_c):
        p = emb_images[classes[x][0]].clone()
        for y in range(1, len(classes[x])):
            p += emb_images[classes[x][y]]
        p /= len(classes[x])
        prototypes.append(p)

    return prototypes

def get_closest_ind(centers, point):
    dists = [torch.dist(point, centers[x]) for x in range(len(centers))]
    min_dist = min(dists)
    min_ind = dists.index(min_dist)
    return min_ind

def get_k_means_centers(emb_images, classes, n_c):
    k_means_centers = get_prototypes(emb_images, classes, n_c)


    for it in range(1):
        closest_centers = [[] for x in range(n_c)]
        
        for x in range(len(emb_images)):
            closest_ind = get_closest_ind(k_means_centers, emb_images[x])
            closest_centers[closest_ind].append(emb_images[x])

        for x in range(n_c):
            center = closest_centers[x][0].clone()
            for y in range(1, len(closest_centers[x])):
                center += closest_centers[x][y]
            center /= len(closest_centers[x])
            k_means_centers[x] = center

    return k_means_centers

def create_smallest_sum_dist_diffs_b(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    classes = [[support_inds_base[x]] for x in range(n_c)]
    support_inds = [ind for ind in support_inds_base]
    for num in range(n_s*n_c - n_c):
        add_diff = 1e30
        add_ind = 0
        prototypes = get_prototypes(emb_images, classes, n_c)

        for x in range(n_t):
            dists = [torch.dist(emb_images[x], prototypes[y]) for y in range(n_c)]
            min_dist = min(dists)
            min_ind = dists.index(min_dist)
            for y in range(n_c):
                dists[y] = dists[y] - min_dist
            dists[min_ind] = max(dists) #don't count it to itself
            min_diff = min(dists)
            if min_diff < add_diff:
                add_diff = min_diff
                add_ind = x

        support_inds.append(add_ind)
        if add_ind < n_c*n_s:
            class_id = add_ind // n_s
        else:
            class_id = (add_ind - n_c * n_s) // n_q
        classes[class_id].append(add_ind)

    query_inds = []
    for x in range(n_t):
        if x not in support_inds:
            query_inds.append(x)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_smallest_sum_dist_diffs_c(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    classes = [[support_inds_base[x]] for x in range(n_c)]
    support_inds = [ind for ind in support_inds_base]
    for num in range(n_s*n_c - n_c):
        add_diff = 1e30
        add_ind = 0
        prototypes = get_k_means_centers(emb_images, classes, n_c)

        for x in range(n_t):
            dists = [torch.dist(emb_images[x], prototypes[y]) for y in range(n_c)]
            min_dist = min(dists)
            min_ind = dists.index(min_dist)
            for y in range(n_c):
                dists[y] = dists[y] - min_dist
            dists[min_ind] = max(dists) #don't count it to itself
            min_diff = min(dists)
            if min_diff < add_diff:
                add_diff = min_diff
                add_ind = x

        support_inds.append(add_ind)
        if add_ind < n_c*n_s:
            class_id = add_ind // n_s
        else:
            class_id = (add_ind - n_c * n_s) // n_q
        classes[class_id].append(add_ind)

    query_inds = []
    for x in range(n_t):
        if x not in support_inds:
            query_inds.append(x)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_smallest_small_dists_unique_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)
    
    b = emb_images[0] + emb_images[1]
    b /=2

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    dists = [1e30 for x in range(n_c)]
    inds = [0 for x in range(n_c)]
    for x in range(n_t):
        class_dists = [torch.dist(emb_images[x], emb_images[support_inds_base[y]]) for y in range(n_c)]
        min_dist = min(class_dists)
        min_ind = class_dists.index(min_dist)
        if (dists[min_ind] > min_dist):
            dists[min_ind] = min_dist
            inds[min_ind] = x

    num_added = n_c
    support_inds = [ind for ind in support_inds_base]
    for ind in inds:
        support_inds.append(ind)
        
    query_inds = []
    for x in range(n_t):
        if x not in support_inds:
            query_inds.append(x)
    
    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)
    
    return new_sample

def create_k_means_centers_batch(sample, model):
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    k_means_centers = get_k_means_centers(emb_images, classes, n_c)

    for x in range(n_c):
        emb_images[support_inds_base[x]] = k_means_centers[x]
        
    query_inds = []
    for x in range(n_t):
        if x not in support_inds:
            query_inds.append(x)

    new_sample= get_new_sample(sample, orig_images, support_inds_base, query_inds, n_c, n_s, n_q)

    return new_sample

def add_one_to_sample(sample, add_ind, model):
    '''Add the single point specified by add_ind'''
    n_c, n_s, n_q, n_t = get_sample_size(sample)
    orig_images = copy_images(sample, n_c, n_s, n_q)
    emb_images = get_embedded_space_images(sample, model)

    support_inds_base = [x * n_s for x in range(n_c)] # add one of each class
    support_inds = [ind for ind in support_inds_base]

    classes = [[] for x in range(n_c)]
    class_counts = [0 for x in range(n_c)]
    for x in support_inds:
        if x < n_c*n_s:
            class_id = x // n_s
        else:
            class_id = (x - n_c * n_s) // n_q
        classes[class_id].append(x)
        class_counts[class_id] += 1

    protos = get_prototypes(emb_images, classes, n_c)
    
    for x in range(n_t):
        if x in support_inds:
            continue
        if add_ind == 0:
            support_inds.append(x)
            break
        add_ind -= 1

    query_inds = []
    for x in range(n_t):
        if x not in support_inds:
            query_inds.append(x)

    new_sample= get_new_sample(sample, orig_images, support_inds, query_inds, n_c, n_s, n_q)

    return new_sample, protos, emb_images[x], class_counts

def get_meta_learning_nn_sample(sample, model):
    best_add = 0
    best_loss_change = -1e30
    input_layer = [tf.feature_column.numeric_column("x", shape=[3925])]

    tf_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,
                                  l2_regularization_strength=0.0)
    #mod_dir = "C:/Users/Cameron/Anaconda2/envs/PY_36/Lib/site-packages/protonets/utils/active_model"
    mod_dir = "active_model"

    voicerecog_classifier = tf.estimator.DNNRegressor(feature_columns=input_layer,
                          hidden_units=[3925,3925], model_dir=mod_dir,
                                      optimizer=tf_optimizer,  dropout=0.5)
    for x in range(95):
        new_sample, protos, counts = create_new_sample(copy.deepcopy(sample), str(x))
        flat_features = compress_protos(protos, counts)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array([flat_features,])},
        y=np.array([0,]),
        num_epochs=1,
        shuffle=False)
        predict_results = voicerecog_classifier.predict(input_fn=predict_input_fn)
        #print(predict_results)
        for item in predict_results:
            val = item['predictions'][0]
            if val > best_loss_change:
                #print(val, x)
                best_add = x
                best_loss_change = val

    new_sample, protos, counts = create_new_sample(copy.deepcopy(sample), best_add)
    return new_sample
    
def create_new_sample(sample, sample_mode, model):
    if sample_mode == "one_shot":
        return create_one_shot_sample(sample, model)
    if sample_mode == "random_equal":
        return create_random_equal_sample(sample, model)
    if sample_mode == "random_unequal":
        return create_random_unequal_sample(sample, model)
    
    if sample_mode == "smallest_small_dists_batch":
        return create_smallest_small_dists_batch(sample, model)
    if sample_mode == "largest_small_dists_batch":
        return create_largest_small_dists_batch(sample, model)
    if sample_mode == "smallest_sum_dists_batch":
        return create_smallest_sum_dists_batch(sample, model)
    if sample_mode == "largest_sum_dists_batch":
        return create_largest_sum_dists_batch(sample, model)
    if sample_mode == "smallest_sum_dist_diffs_batch":
        return create_smallest_sum_dist_diffs_batch(sample, model)
    if sample_mode == "largest_sum_dist_diffs_batch":
        return create_largest_sum_dist_diffs_batch(sample, model)
    
    if sample_mode == "smallest_sum_dist_diffs_pair_batch":
        return create_smallest_sum_dist_diffs_pair_batch(sample, model)
    if sample_mode == "largest_sum_dist_diffs_pair_batch":
        return create_largest_sum_dist_diffs_pair_batch(sample, model)
    if sample_mode == "smallest_sum_dist_diffs1_rand_batch":
        return create_smallest_sum_dist_diffs1_rand_batch(sample, model)
    if sample_mode == "smallest_small_dists_unique_batch":
        return create_smallest_small_dists_unique_batch(sample, model)

    if sample_mode == "k_means_centers_batch":
        return create_k_means_centers_batch(sample, model)
    
    if sample_mode == "smallest_sum_dist_diffs":
        return create_smallest_sum_dist_diffs(sample, model)
    if sample_mode == "smallest_sum_dist_diffs_b":
        return create_smallest_sum_dist_diffs_b(sample, model)
    if sample_mode == "smallest_sum_dist_diffs_c":
        return create_smallest_sum_dist_diffs_c(sample, model)
    if sample_mode == "meta_learning_nn":
        return get_meta_learning_nn_sample(sample, model)
    return add_one_to_sample(sample, int(sample_mode), model)
    raise ValueError
    return sample
