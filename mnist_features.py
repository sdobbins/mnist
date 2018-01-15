# @author Scott Dobbins
# @version 0.1
# @date 2017-12-02

import numpy as np
from skimage import measure as meas
from skimage import morphology as morph

def ridge_detector(img, grad_r = 1.5, search_r = 2.5):
    check_radius(grad_r)
    check_radius(search_r)
    # make sure large (including negative) differences won't overflow
    if img.dtype != np.float_:
        img = np.array(img, dtype = np.float_)
    grad = gradient(img, grad_r)
    slicer = dists <= search_r
    #slicer = np.logical_and(dists <= search_r, np.logical_or(dists_x > 0, np.logical_and(dists_x == 0, dists_y > 0)))
    inv_dists_r = inv_dists[slicer]
    inv_dists_r_sq = np.square(inv_dists_r)
    dirs_r = dirs[slicer]
    conv = np.zeros(img.shape + (len(dirs_r),))
    for d in range(len(dirs_r)):
        for p in range(len(dirs_r)):
            conv[:,:,d] += np.cross(dirs_r[d], shifted_image(grad, dirs_r[p])) * np.cross(dirs_r[p], dirs_r[d]) * inv_dists_r_sq[d] * inv_dists_r[p]
    return conv

def best_args(img, thresh = 0.1, grad_r = 1.5, search_r = 2.5):
    best = np.argmax(ridge_detector(img, grad_r, search_r), axis = -1)
    best[img < thresh] = -1
    return best

def bordering_objects(obj_labels, obj):
    is_in_obj = np.equal(obj_labels, obj)
    is_in_obj_borders = morph.dilation(is_in_obj, selem = morph.square(3))
    bordering = np.unique(obj_labels[is_in_obj_borders])
    return bordering[np.logical_and(np.not_equal(bordering, 0), np.not_equal(bordering, obj))]

def bordering_objects_by_extent(obj_labels, obj):
    is_in_obj = np.equal(obj_labels, obj)
    is_in_obj_borders = morph.dilation(is_in_obj, selem = morph.square(3))
    border_labels = obj_labels[is_in_obj_borders]
    bordering = np.unique(border_labels)
    bordering = bordering[np.logical_and(np.not_equal(bordering, 0), np.not_equal(bordering, obj))]
    border_counts = {bord:(np.sum(np.equal(border_labels, bord))) for bord in bordering}
    return border_counts

def fix_args(img, thresh = 0.1, grad_r = 1.5, search_r = 2.5, test_r = 1.5):
    dirs_r = dirs[dists <= search_r]
    args = best_args(img, thresh, grad_r, search_r)
    flow = dirs_r[args]
    flow[np.equal(args, -1)] = (0,0)
    clashes = test_flow_dir(flow, thresh, test_r)
    total_clash_intensity = np.sum(clashes)
    object_labels = morph.label(args, background = -1, connectivity = 2)
    all_objects = np.unique(object_labels)
    all_objects = all_objects[np.not_equal(all_objects, 0)]
    object_args = {obj:(args[tuple(np.argwhere(np.equal(object_labels, obj))[0])]) for obj in all_objects}
    object_flows = {obj:(flow[tuple(np.argwhere(np.equal(object_labels, obj))[0])]) for obj in all_objects}
    continent_args = np.where(args == -1, -1, 0)
    continent_labels = morph.label(continent_args, background = -1, connectivity = 2)
    all_continents = np.unique(continent_labels)
    all_continents = all_continents[np.not_equal(all_continents, 0)]
    #num_continents = np.max(all_continents)
    for continent in all_continents:
        pixels_this_continent = object_labels[np.equal(continent_labels, continent)]
        objects_in_continent = np.unique(pixels_this_continent)
        object_sizes = {obj:0 for obj in objects_in_continent}
        for o in range(len(pixels_this_continent)):
            object_sizes[pixels_this_continent[o]] += 1
        biggest_object = max(object_sizes, key = object_sizes.get)
        continent_object_labels = np.copy(object_labels)
        continent_object_labels[np.isin(continent_object_labels, objects_in_continent, invert = True)] = 0
        #remaining_other_objects = objects_in_continent[np.not_equal(objects_in_continent, biggest_object)]
        border_objects = bordering_objects(continent_object_labels, biggest_object)
        while len(border_objects) > 0:
            clash_labels = continent_object_labels[clashes > 4]#*** consider setting this differently
            #clash_labels = clash_labels[np.logical_and(np.not_equal(clash_labels, 0), np.not_equal(clash_labels, biggest_object))]
            clash_objects = np.unique(clash_labels)
            problem_objects = np.intersect1d(border_objects, clash_objects)
#            clash_sizes = {obj:0 for obj in all_objects}
#            for obj in clash_labels:
#                clash_sizes[obj] += 1
            for obj in problem_objects:
                temp_flow = np.copy(flow)
                temp_flow[np.equal(continent_object_labels, obj)] = -object_flows[obj]
                temp_clashes = test_flow_dir(temp_flow, thresh, test_r)
                temp_clash_intensity = np.sum(temp_clashes)
                if temp_clash_intensity < total_clash_intensity:
                    total_clash_intensity = temp_clash_intensity
                    new_flow = -object_flows[obj]
                    new_arg = np.argwhere(np.logical_and(np.equal(dirs_r[:,0], new_flow[0]), np.equal(dirs_r[:,1], new_flow[1]))).item()
                    flow[np.equal(continent_object_labels, obj)] = new_flow
                    args[np.equal(continent_object_labels, obj)] = new_arg
                    object_flows[obj] = new_flow
                    object_args[obj] = new_arg
            for obj in border_objects:
                continent_object_labels[np.equal(continent_object_labels, obj)] = biggest_object
                #remaining_other_objects = remaining_other_objects[np.not_equal(remaining_other_objects, obj)]
            border_objects = bordering_objects(continent_object_labels, biggest_object)
    return args
                
def join_objects_incomplete(args, search_r = 2.5):
    small_object_size_limit = 4 #*** how to best choose this?
    remaining_object_limit = 1 #*** how to best choose this?
    
    object_labels = morph.label(args, background = -1, connectivity = 2)
    
    all_objects = np.unique(object_labels)
    all_objects = all_objects[np.not_equal(all_objects, 0)]
    
    #dirs_r = dirs[dists <= search_r]
    normed_dirs_r = normed_dirs[dists <= search_r]
    #flow = dirs_r[args]
    #flow[np.equal(args, -1)] = (0,0)
    flow_dir = normed_dirs_r[args]
    flow_dir[np.equal(args, -1)] = (0,0)
    
    #object_args = {obj:(args[tuple(np.argwhere(np.equal(object_labels, obj))[0])]) for obj in all_objects}
    #object_flows = {obj:(flow[tuple(np.argwhere(np.equal(object_labels, obj))[0])]) for obj in all_objects}
    object_flow_dirs = {obj:(flow_dir[tuple(np.argwhere(np.equal(object_labels, obj))[0])]) for obj in all_objects}
    
    continent_args = np.where(args == -1, -1, 0)
    continent_labels = morph.label(continent_args, background = -1, connectivity = 2)
    all_continents = np.unique(continent_labels)
    all_continents = all_continents[np.not_equal(all_continents, 0)]
    
    for continent in all_continents:
        pixels_this_continent = object_labels[np.equal(continent_labels, continent)]
        objects_in_continent = np.unique(pixels_this_continent)
        if len(objects_in_continent) > 1:
            object_sizes = {obj:0 for obj in objects_in_continent}
            for o in range(len(pixels_this_continent)):
                object_sizes[pixels_this_continent[o]] += 1
            
            smallest_object = min(object_sizes, key = object_sizes.get)
            smallest_object_size = object_sizes[smallest_object]
            
            while smallest_object_size < small_object_size_limit:
                bordering_counts = bordering_objects_by_extent(object_labels, smallest_object)
                
                largest_overlap_object = max(bordering_counts, key = bordering_counts.get)
                largest_overlap = bordering_counts[largest_overlap_object]
                all_tied_objects = np.array([thing for thing in bordering_counts.keys() if bordering_counts[thing] >= largest_overlap-1])
                if len(all_tied_objects) == 1:
                    best_object = all_tied_objects[0].item()
                else:
                    tied_flows = np.array([object_flow_dirs[obj] for obj in all_tied_objects])
                    tied_dots = np.inner(tied_flows, object_flow_dirs[smallest_object])
                    best_dot = np.max(tied_dots)
                    all_tied_objects = all_tied_objects[np.equal(tied_dots, best_dot)]
                    if len(all_tied_objects) == 1:
                        best_object = all_tied_objects[0].item()
                    else:
                        region_props = meas.regionprops(object_labels)
                        this_centroid = np.array(region_props[np.where(np.equal(all_objects, smallest_object))[0].item()]['centroid'])
                        other_centroids = np.array([region_props[np.where(np.equal(all_objects, obj))[0].item()]['centroid'] for obj in all_tied_objects])
                        tied_dists = np.linalg.norm(other_centroids - this_centroid, axis = 1)
                        best_dist = np.min(tied_dists)
                        best_object = all_tied_objects[np.equal(tied_dists, best_dist)][0].item()
                        #*** also put in one more step whereby if dists are the same, it can pick the one in the direction of its flow--necessary or not???
                object_labels[np.equal(object_labels, smallest_object)] = best_object
                #object_flow_dirs[best_object] = (object_flow_dirs[best_object] * object_sizes[best_object] + object_flow_dirs[smallest_object] * object_sizes[smallest_object]) / (object_sizes[best_object] + object_sizes[smallest_object])
                object_sizes[best_object] += object_sizes.pop(smallest_object)
                all_objects = all_objects[np.not_equal(all_objects, smallest_object)]
                objects_in_continent = objects_in_continent[np.not_equal(objects_in_continent, smallest_object)]
                if len(object_sizes) > 0:
                    smallest_object = min(object_sizes, key = object_sizes.get)
                    smallest_object_size = object_sizes[smallest_object]
                else:
                    smallest_object_size = small_object_size_limit+1
    return object_labels

def join_objects(args, search_r = 2.5):
    small_object_size_limit = 4 #*** how to best choose this?
    remaining_object_limit = 1 #*** how to best choose this?
    
    object_labels = morph.label(args, background = -1, connectivity = 2)
    
    all_objects = np.unique(object_labels)
    all_objects = all_objects[np.not_equal(all_objects, 0)]
    
    #dirs_r = dirs[dists <= search_r]
    normed_dirs_r = normed_dirs[dists <= search_r]
    #flow = dirs_r[args]
    #flow[np.equal(args, -1)] = (0,0)
    flow_dir = normed_dirs_r[args]
    flow_dir[np.equal(args, -1)] = (0,0)
    
    #object_args = {obj:(args[tuple(np.argwhere(np.equal(object_labels, obj))[0])]) for obj in all_objects}
    #object_flows = {obj:(flow[tuple(np.argwhere(np.equal(object_labels, obj))[0])]) for obj in all_objects}
    object_flow_dirs = {obj:(flow_dir[tuple(np.argwhere(np.equal(object_labels, obj))[0])]) for obj in all_objects}
    
    continent_args = np.where(args == -1, -1, 0)
    continent_labels = morph.label(continent_args, background = -1, connectivity = 2)
    all_continents = np.unique(continent_labels)
    all_continents = all_continents[np.not_equal(all_continents, 0)]
    
    for continent in all_continents:
        pixels_this_continent = object_labels[np.equal(continent_labels, continent)]
        objects_in_continent = np.unique(pixels_this_continent)
        if len(objects_in_continent) > 1:
            object_sizes = {obj:0 for obj in objects_in_continent}
            for o in range(len(pixels_this_continent)):
                object_sizes[pixels_this_continent[o]] += 1
            
            smallest_object = min(object_sizes, key = object_sizes.get)
            smallest_object_size = object_sizes[smallest_object]
            
            while smallest_object_size < small_object_size_limit:
                bordering_counts = bordering_objects_by_extent(object_labels, smallest_object)
                
                largest_overlap_object = max(bordering_counts, key = bordering_counts.get)
                largest_overlap = bordering_counts[largest_overlap_object]
                all_tied_objects = np.array([thing for thing in bordering_counts.keys() if bordering_counts[thing] >= largest_overlap-1])
                if len(all_tied_objects) == 1:
                    best_object = all_tied_objects[0].item()
                else:
                    tied_flows = np.array([object_flow_dirs[obj] for obj in all_tied_objects])
                    tied_dots = np.inner(tied_flows, object_flow_dirs[smallest_object])
                    best_dot = np.max(tied_dots)
                    all_tied_objects = all_tied_objects[np.equal(tied_dots, best_dot)]
                    if len(all_tied_objects) == 1:
                        best_object = all_tied_objects[0].item()
                    else:
                        region_props = meas.regionprops(object_labels)
                        this_centroid = np.array(region_props[np.where(np.equal(all_objects, smallest_object))[0].item()]['centroid'])
                        other_centroids = np.array([region_props[np.where(np.equal(all_objects, obj))[0].item()]['centroid'] for obj in all_tied_objects])
                        tied_dists = np.linalg.norm(other_centroids - this_centroid, axis = 1)
                        best_dist = np.min(tied_dists)
                        best_object = all_tied_objects[np.equal(tied_dists, best_dist)][0].item()
                        #*** also put in one more step whereby if dists are the same, it can pick the one in the direction of its flow--necessary or not???
                object_labels[np.equal(object_labels, smallest_object)] = best_object
                #object_flow_dirs[best_object] = (object_flow_dirs[best_object] * object_sizes[best_object] + object_flow_dirs[smallest_object] * object_sizes[smallest_object]) / (object_sizes[best_object] + object_sizes[smallest_object])
                object_sizes[best_object] += object_sizes.pop(smallest_object)
                all_objects = all_objects[np.not_equal(all_objects, smallest_object)]
                objects_in_continent = objects_in_continent[np.not_equal(objects_in_continent, smallest_object)]
                if len(object_sizes) > 0:
                    smallest_object = min(object_sizes, key = object_sizes.get)
                    smallest_object_size = object_sizes[smallest_object]
                else:
                    smallest_object_size = small_object_size_limit+1
            
            remaining_objects = np.copy(objects_in_continent)
            remaining_object_sizes = object_sizes.copy()
            
            while len(remaining_objects) > remaining_object_limit:
                bordering_counts = bordering_objects_by_extent(object_labels, smallest_object)
                
                largest_overlap_object = max(bordering_counts, key = bordering_counts.get)
                largest_overlap = bordering_counts[largest_overlap_object]
                all_tied_objects = np.array([thing for thing in bordering_counts.keys() if bordering_counts[thing] >= largest_overlap / 2])
                
                tied_flows = np.array([object_flow_dirs[obj] for obj in all_tied_objects])
                tied_dots = np.inner(tied_flows, object_flow_dirs[smallest_object])
                
                all_tied_objects = all_tied_objects[np.greater(tied_dots, 0.0)]
                
                if len(all_tied_objects) == 1:
                    best_object = all_tied_objects[0].item()
                elif len(all_tied_objects) > 1:
                    tied_dots = tied_dots[np.greater(tied_dots, 0.0)]
                    best_dot = np.max(tied_dots)
                    all_tied_objects = all_tied_objects[np.equal(tied_dots, best_dot)]
                    if len(all_tied_objects) == 1:
                        best_object = all_tied_objects[0].item()
                    else:
                        region_props = meas.regionprops(object_labels)
                        this_centroid = np.array(region_props[np.where(np.equal(all_objects, smallest_object))[0].item()]['centroid'])
                        other_centroids = np.array([region_props[np.where(np.equal(all_objects, obj))[0].item()]['centroid'] for obj in all_tied_objects])
                        tied_dists = np.linalg.norm(other_centroids - this_centroid, axis = 1)
                        best_dist = np.max(tied_dists)
                        best_object = all_tied_objects[np.equal(tied_dists, best_dist)].item()
                else:
                    best_object = None
                
                if best_object is not None:
                    object_labels[np.equal(object_labels, smallest_object)] = best_object
                    #object_flow_dirs[best_object] = (object_flow_dirs[best_object] * object_sizes[best_object] + object_flow_dirs[smallest_object] * object_sizes[smallest_object]) / (object_sizes[best_object] + object_sizes[smallest_object])
                    object_sizes[best_object] += object_sizes.pop(smallest_object)
                    all_objects = all_objects[np.not_equal(all_objects, smallest_object)]
                    objects_in_continent = objects_in_continent[np.not_equal(objects_in_continent, smallest_object)]
                remaining_object_sizes.pop(smallest_object)
                remaining_objects = remaining_objects[np.not_equal(remaining_objects, smallest_object)]
                smallest_object = min(remaining_object_sizes, key = remaining_object_sizes.get)
    return object_labels
