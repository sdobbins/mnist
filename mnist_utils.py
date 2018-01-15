# @author Scott Dobbins
# @version 0.1
# @date 2017-12-02

import numpy as np

### start analysis

# gradients
biggest_rad = 3
rad_size = biggest_rad * 2 + 1

dists_x, dists_y = np.indices([biggest_rad*2+1]*2) - biggest_rad
# large dist for (0,0) so that (0,0) is not included in any calcluations
dists_x[biggest_rad, biggest_rad] = 1<<16
dists_y[biggest_rad, biggest_rad] = 1<<16

dirs = np.stack((dists_x, dists_y), axis = -1)

normed_dirs = dirs / np.linalg.norm(dirs, axis = -1).reshape(dirs.shape[:2] + (1,))

dists = np.linalg.norm(np.stack([dists_x, dists_y]), axis = 0)

inv_dists = 1.0 / dists

are_int_dists = dists == dists.round()

floor_dists = np.floor(dists).astype(int)
int_dists = np.rint(dists).astype(int)
ceil_dists = np.ceil(dists).astype(int)

### angle functions

def dir_angle(d):
    if np.isclose(d[0], 0):
        if np.isclose(d[1], 0):
            return 0.0
        else:
            if d[1] > 0:
                return np.pi / 2.0
            else:
                return -np.pi / 2.0
    else:
        if d[0] < 0:
            if d[1] < 0:
                return -np.pi + np.arctan(d[1] / d[0])
            else:
                return np.pi + np.arctan(d[1] / d[0])
        else:
            return np.arctan(d[1] / d[0])

def fix_angle(a):
    if a > np.pi:
        return a - 2.0 * np.pi
    elif a < -np.pi:
        return 2.0 * np.pi + a
    elif np.isclose(a, -np.pi):
        return np.pi
    else:
        return a

def angle_between_dirs(d1, d2):
    angle = dir_angle(d2) - dir_angle(d1)
    return fix_angle(angle)

def angle_between_angles(a1, a2):
    angle = a2 - a1
    return fix_angle(angle)

def diff_from_linear(d1, d2):
    angle = angle_between_dirs(d1, d2)
    if np.isclose(np.abs(angle), -np.pi):
        return 0.0
    else:
        if angle < 0:
            return angle + np.pi
        else:
            return angle - np.pi

def angle_set(ds):
    if type(ds) == np.ndarray:
        angles = np.zeros(ds.shape[0])
    else:
        angles = [0] * len(ds)
    for d in range(len(ds)):
        angles[d] = dir_angle(ds[d])
    return angles

angles = np.zeros(dists.shape)
for i in range(-biggest_rad, biggest_rad+1):
    for j in range(-biggest_rad, biggest_rad+1):
        angles[i,j] = dir_angle(dirs[i,j])

### other functions

def check_radius(r):
    if r >= biggest_rad + 1:
        raise ValueError('gradient(img, r, d) not implemented for r >= biggest_rad + 1')
    elif r < 1:
        raise ValueError('r must be >= 1')

def check_threshold(t):
    if t <= 0.0 or t >= 1:
        raise ValueError('gradient threshold must be in (0,1)')

def shifted_image(img, d, axes = (0,1), zerofill = False):
    # provide x indices or y indices for shifting img array
    # last single row or column included is repeated in padding
    # so as to make gradient = 0 along border (unless zerofill = True)
    x_axis = axes[0]
    y_axis = axes[1]
    if d[0] < 0:
        clipped_x_inds = [-d[0]-1] * -d[0]
        x_inds = clipped_x_inds + range(img.shape[x_axis] + d[0])
    elif d[0] > 0:
        clipped_x_inds = [img.shape[x_axis] - d[0]] * d[0]
        x_inds = range(d[0], img.shape[x_axis]) + clipped_x_inds
    if d[1] < 0:
        clipped_y_inds = [-d[1]-1] * -d[1]
        y_inds = clipped_y_inds + range(img.shape[y_axis] + d[1])
    elif d[1] > 0:
        clipped_y_inds = [img.shape[y_axis] - d[1]] * d[1]
        y_inds = range(d[1], img.shape[y_axis]) + clipped_y_inds
    # prepare to return simple gradient-like subtraction using shifted array
    if d[0] == 0:
        if d[1] == 0:
            temp_img = img
        else:
            temp_img = img.take(y_inds, y_axis)
    else:
        if d[1] == 0:
            temp_img = img.take(x_inds, x_axis)
        else:
            temp_img = img.take(x_inds, x_axis).take(y_inds, y_axis)
    # if desired, pad with 0s
    if zerofill:
        indices = np.indices(np.array(img.shape)[list(axes)])
        if d[0] < 0:
            np.place(temp_img, vals = 0, mask = indices[0] <= clipped_x_inds[0])
        elif d[0] > 0:
            np.place(temp_img, vals = 0, mask = indices[0] >= clipped_x_inds[0])
        if d[1] < 0:
            np.place(temp_img, vals = 0, mask = indices[1] <= clipped_y_inds[0])
        elif d[1] > 0:
            np.place(temp_img, vals = 0, mask = indices[1] >= clipped_y_inds[0])
    return temp_img

def gradient_part(img, d):
    # make sure large (including negative) differences won't overflow
    if img.dtype != np.float_:
        img = np.array(img, dtype = np.float_)
    if tuple(d) == (0,0): # faster form of easy case
        return np.zeros(img.shape, img.dtype)
    else:
        return shifted_image(img, d) - img

def normalize(arr, ord = None, axis = None, keepdims = False):
    norm = np.linalg.norm(arr, ord, axis, keepdims)
    if norm.ndim == 1:
        if norm == 0:
            return arr
        else:
            return arr / norm
    else:
        not_0 = norm != 0
        new_arr = np.copy(arr)
        new_arr[not_0] /= norm[not_0].reshape((-1,1))
        return new_arr

def gradient_x(img, r = 1.5):
    check_radius(r)
    # make sure large (including negative) differences won't overflow
    if img.dtype != np.float_:
        img = np.array(img, dtype = np.float_)
    slicer = np.logical_and(dists <= r, dists_x != 0)
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    signs = np.sign(dists_x[slicer])
    grad = np.zeros(img.shape, img.dtype)
    for d in range(len(dirs_r)):
        grad += gradient_part(img, dirs_r[d]) * inv_dists_r[d] * signs[d]
    return grad / np.sum(inv_dists_r)

def gradient_y(img, r = 1.5):
    check_radius(r)
    # make sure large (including negative) differences won't overflow
    if img.dtype != np.float_:
        img = np.array(img, dtype = np.float_)
    slicer = np.logical_and(dists <= r, dists_y != 0)
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    signs = np.sign(dists_y[slicer])
    grad = np.zeros(img.shape, img.dtype)
    for d in range(len(dirs_r)):
        grad += gradient_part(img, dirs_r[d]) * inv_dists_r[d] * signs[d]
    return grad / np.sum(inv_dists_r)

def gradient(img, r = 1.5):
    return np.stack([gradient_x(img, r), gradient_y(img, r)], axis = -1)

def avg(img, r = 1.5):
    check_radius(r)
    slicer = dists <= r
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    avg = np.zeros(img.shape)
    for d in range(len(dirs_r)):
        avg += shifted_image(img, dirs_r[d]) * inv_dists_r[d]
    return avg / sum(inv_dists_r)

def gradient_mag(img, r = 1.5):
    return np.linalg.norm(gradient(img, r), axis = -1)

def gradient_dir(img, r = 1.5):
    return normalize(gradient(img, r), axis = -1)

def peakness(img, r = 1.5):
    check_radius(r)
    # make sure large (including negative) differences won't overflow
    if img.dtype != np.float_:
        img = np.array(img, dtype = np.float_)
    slicer = dists <= r
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    peak = np.zeros(img.shape, img.dtype)
    for d in range(len(dirs_r)):
        peak -= gradient_part(img, dirs_r[d]) * inv_dists_r[d]
    return peak / sum(inv_dists_r)

def convergence(img, r = 1.5):
    check_radius(r)
    # make sure large (including negative) differences won't overflow
    if img.dtype != np.float_:
        img = np.array(img, dtype = np.float_)
    grad = gradient(img, r)
    slicer = dists <= r
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    converge = np.zeros(img.shape, img.dtype)
    for d in range(len(dirs_r)):
        converge += np.inner(shifted_image(grad, dirs_r[d]), -dirs_r[d]) * inv_dists_r[d]
    return converge / sum(inv_dists_r)

def focus_pixels(img, r = 1.5, grad_thresh = 0.01, curve_r_range = None):
    check_threshold(grad_thresh)
    img_bound_x = img.shape[0] - 2 # 2 because we need to protect the inhibiory side tracks, which stick out an extra pixel
    img_bound_y = img.shape[1] - 2 # 2 because we need to protect the inhibiory side tracks, which stick out an extra pixel
    if curve_r_range is None:
        curve_r_range = (1, max(img.shape) - 2) # 2 because we need to protect the inhibiory side tracks, which stick out an extra pixel
    grad = gradient(img, r)
    grad_x = grad[:,:,0]
    grad_y = grad[:,:,1]
    mag = np.linalg.norm(grad, axis = -1)
    not_0 = mag != 0
    grad_dir_x = np.copy(grad_x)
    grad_dir_x[not_0] /= mag[not_0]
    grad_dir_y = np.copy(grad_y)
    grad_dir_y[not_0] /= mag[not_0]
    threshold = grad_thresh * np.max(mag)
    thresholded = mag >= threshold
    xs, ys = np.indices(img.shape)
    xs_thresh = xs[thresholded]
    ys_thresh = ys[thresholded]
    curves_result = np.zeros((img.shape + (curve_r_range[1] + 1, 2)))
    for d in range(len(xs_thresh)):
        # figure out how long lines will be
        orig_x = xs_thresh[d]
        orig_y = ys_thresh[d]
        orig_pos = (orig_x, orig_y)
        orig_grad_x = grad_dir_x[orig_pos]
        if orig_grad_x == 0:
            x_len_p = curve_r_range[1]
            x_len_n = curve_r_range[1]
        elif orig_grad_x > 0:
            x_len_p = int(((img_bound_x - orig_x) / orig_grad_x).round())
            x_len_n = int((orig_x / orig_grad_x).round())
        else:
            x_len_p = int((orig_x / -orig_grad_x).round())
            x_len_n = int(((img_bound_x - orig_x) / -orig_grad_x).round())
        orig_grad_y = grad_dir_y[orig_pos]
        if orig_grad_y == 0:
            y_len_p = curve_r_range[1]
            y_len_n = curve_r_range[1]
        elif orig_grad_y > 0:
            y_len_p = int(((img_bound_y - orig_y) / orig_grad_y).round())
            y_len_n = int((orig_y / orig_grad_y).round())
        else:
            y_len_p = int((orig_y / -orig_grad_y).round())
            y_len_n = int(((img_bound_y - orig_y) / -orig_grad_y).round())
        length_p = min([x_len_p, y_len_p, curve_r_range[1]])
        length_n = min([x_len_n, y_len_n, curve_r_range[1]])
        # prepare to spread information
        side_track_1, side_track_2 = side_tracks((orig_grad_x, orig_grad_y))
        # spread info on positive side
        c = np.arange(curve_r_range[0], length_p + 1)
        amounts = mag[orig_pos] / c # (as normalizer instead of separate step below)
        pos_x = orig_x + int((c * orig_grad_x).round())
        pos_y = orig_y + int((c * orig_grad_y).round())
        curves_result[pos_x, pos_y, c, 0] += 2 * amounts
        curves_result[pos_x + side_track_1[0], pos_y + side_track_1[1], c, 0] -= amounts
        curves_result[pos_x + side_track_2[0], pos_y + side_track_2[1], c, 0] -= amounts
        # spread info on negative side
        c = np.arange(curve_r_range[0], length_n + 1)
        amounts = mag[orig_pos] / c # (as normalizer instead of separate step below)
        pos_x = orig_x - int((c * orig_grad_x).round())
        pos_y = orig_y - int((c * orig_grad_y).round())
        curves_result[pos_x, pos_y, c, 1] += 2 * amounts
        curves_result[pos_x + side_track_1[0], pos_y + side_track_1[1], c, 1] -= amounts
        curves_result[pos_x + side_track_2[0], pos_y + side_track_2[1], c, 1] -= amounts
#    normalizer = np.arange(curves_result.shape[2]).reshape((1,1,curves_result.shape[2],1))
#    curves_result[:,:,curve_r_range[0]:,:] /= normalizer[:,:,curve_r_range[0]:,:]
    curves_result[:,:,0,:] = np.argmax(curves_result, axis = 2)
    return curves_result

def focus_pixels_no_interference(img, r = 1.5, grad_thresh = 0.01, curve_r_range = None):
    check_threshold(grad_thresh)
    img_max_x = img.shape[0] - 1
    img_max_y = img.shape[1] - 1
    img_max_len = max(img.shape) - 1
    if curve_r_range is None:
        curve_r_range = (1, img_max_len)
    grad = gradient(img, r)
    grad_x = grad[:,:,0]
    grad_y = grad[:,:,1]
    mag = np.linalg.norm(grad, axis = -1)
    not_0 = mag != 0
    grad_dir_x = np.copy(grad_x)
    grad_dir_x[not_0] /= mag[not_0]
    grad_dir_y = np.copy(grad_y)
    grad_dir_y[not_0] /= mag[not_0]
    threshold = grad_thresh * np.max(mag)
    thresholded = mag >= threshold
    xs, ys = np.indices(img.shape)
    xs_thresh = xs[thresholded]
    ys_thresh = ys[thresholded]
    curves_result = np.zeros((img.shape + (curve_r_range[1] + 1, 2)))
    for d in range(len(xs_thresh)):#*** see if you can vectorize (at least part of) this process
        # figure out how long lines will be
        orig_x = xs_thresh[d]
        orig_y = ys_thresh[d]
        orig_pos = (orig_x, orig_y)
        orig_grad_x = grad_dir_x[orig_pos]
        if orig_grad_x == 0:
            x_len_p = curve_r_range[1]
            x_len_n = curve_r_range[1]
        elif orig_grad_x > 0:
            x_len_p = int(((img_max_x - orig_x) / orig_grad_x).round())
            x_len_n = int((orig_x / orig_grad_x).round())
        else:
            x_len_p = int((orig_x / -orig_grad_x).round())
            x_len_n = int(((img_max_x - orig_x) / -orig_grad_x).round())
        orig_grad_y = grad_dir_y[orig_pos]
        if orig_grad_y == 0:
            y_len_p = curve_r_range[1]
            y_len_n = curve_r_range[1]
        elif orig_grad_y > 0:
            y_len_p = int(((img_max_y - orig_y) / orig_grad_y).round())
            y_len_n = int((orig_y / orig_grad_y).round())
        else:
            y_len_p = int((orig_y / -orig_grad_y).round())
            y_len_n = int(((img_max_y - orig_y) / -orig_grad_y).round())
        length_p = min([x_len_p, y_len_p, curve_r_range[1]])
        length_n = min([x_len_n, y_len_n, curve_r_range[1]])
        # spread information
        c = np.arange(curve_r_range[0], length_p + 1)
        pos_x = orig_x + int((c * orig_grad_x).round())
        pos_y = orig_y + int((c * orig_grad_y).round())
        curves_result[pos_x, pos_y, c, 0] += 2 * mag[orig_pos] / c #(as normalizer instead of separate step below)
        c = np.arange(curve_r_range[0], length_n + 1)
        pos_x = orig_x - int((c * orig_grad_x).round())
        pos_y = orig_y - int((c * orig_grad_y).round())
        curves_result[pos_x, pos_y, c, 1] += 2 * mag[orig_pos] / c #(as normalizer instead of separate step below)
#    normalizer = np.arange(curves_result.shape[2]).reshape((1,1,curves_result.shape[2],1))
#    curves_result[:,:,curve_r_range[0]:,:] /= normalizer[:,:,curve_r_range[0]:,:]
    curves_result[:,:,0,:] = np.argmax(curves_result, axis = 2)
    return curves_result

def orthogonal_dirs(d):
    result = [[d[1], -d[0]], 
              [-d[1], d[0]]]
    if type(d) == np.ndarray:
        return np.array(result)
    else:
        return result

def multi_orthogonal_dirs(ds):
    if type(ds) == np.ndarray:
        results = np.zeros((ds.shape[0],) + (2,2), dtype = ds.dtype)
    else:
        results = [0] * len(ds)
    for d in range(len(ds)):
        results[d] = orthogonal_dirs(ds[d])
    return results

def side_tracks(d):
    rounded = d.round()
    if any(rounded == 0):
        return orthogonal_dirs(rounded)
    else:
        if rounded[0] < rounded[1]:
            return orthogonal_dirs(np.array([0, rounded[1]]))
        else:
            return orthogonal_dirs(np.array([rounded[0], 0]))

def concavity(img, grad_r = 1.5, concav_r = 2.5):
    check_radius(concav_r)
    grad_dir = gradient_dir(img, grad_r)
    slicer = dists <= concav_r
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    inv_dists_r_sq = np.square(inv_dists_r)
    concav = np.zeros(img.shape, img.dtype)
    for d in range(len(dirs_r)):
        concav += np.cross(grad_dir, shifted_image(grad_dir, dirs_r[d])) * np.cross(dirs_r[d], grad_dir) * inv_dists_r_sq[d]
    return concav / sum(inv_dists_r)

def concavity_avg(img, grad_r = 1.5, concav_r = 2.5, avg_r = 2.5):
    check_radius(concav_r)
    check_radius(avg_r)
    grad_dir = gradient_dir(img, grad_r)
    avg_grad_dir = avg(grad_dir, avg_r)
    slicer = dists <= concav_r
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    inv_dists_r_sq = np.square(inv_dists_r)
    concav = np.zeros(img.shape, img.dtype)
    for d in range(len(dirs_r)):
        concav += np.cross(avg_grad_dir, shifted_image(grad_dir, dirs_r[d])) * np.cross(dirs_r[d], avg_grad_dir) * inv_dists_r_sq[d]
    return concav / sum(inv_dists_r)

def concavity_proportional(img, grad_r = 1.5, concav_r = 2.5):
    check_radius(concav_r)
    grad = gradient(img, grad_r)
    mag = np.linalg.norm(grad, axis = -1)
    grad_dir = np.copy(grad)
    not_0 = mag != 0
    grad_dir[not_0] /= mag[not_0].reshape((-1,1))
    slicer = dists <= concav_r
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    inv_dists_r_sq = np.square(inv_dists_r)
    concav = np.zeros(img.shape, img.dtype)
    for d in range(len(dirs_r)):
        concav += np.cross(grad_dir, shifted_image(grad_dir, dirs_r[d])) * np.cross(dirs_r[d], grad) * inv_dists_r_sq[d]
    return concav / sum(inv_dists_r)

# should concavity be weighted toward larger mags or not? (divide by mag immediately for each section or divide total by total)

def concavity_proportional_avg(img, grad_r = 1.5, concav_r = 2.5, avg_r = 2.5):
    check_radius(concav_r)
    check_radius(avg_r)
    grad = gradient(img, grad_r)
    avg_grad = avg(grad, avg_r)
    mag = np.linalg.norm(avg_grad, axis = -1)
    avg_grad_dir = np.copy(avg_grad)
    not_0 = mag != 0
    avg_grad_dir[not_0] /= mag[not_0].reshape((-1,1))
    slicer = dists <= concav_r
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    inv_dists_r_sq = np.square(inv_dists_r)
    concav = np.zeros(img.shape, img.dtype)
    for d in range(len(dirs_r)):
        concav += np.cross(avg_grad_dir, shifted_image(grad, dirs_r[d])) * np.cross(dirs_r[d], avg_grad_dir) * inv_dists_r_sq[d]
    return concav / sum(inv_dists_r)

def aligned(img, r = 1.5):
    grad_dir = gradient_dir(img, r)
    slicer = dists <= r
    dirs_r = dirs[slicer]
    inv_dists_r = inv_dists[slicer]
    inv_dists_r_sq = np.square(inv_dists_r)
    align = np.zeros(img.shape, img.dtype)
    for d in range(len(dirs_r)):
        align += np.sum(grad_dir * shifted_image(grad_dir, dirs_r[d]), axis = -1) * np.abs(np.cross(dirs_r[d], grad_dir)) * inv_dists_r_sq[d]
    return align / sum(inv_dists_r)

def line_pixels(img, r = 1.5, grad_thresh = 0.01, contig_thresh = 0.5):
    img_max_x = img.shape[0] - 1
    img_max_y = img.shape[1] - 1
    img_max_len = max(img.shape) - 1
    grad = gradient(img, r)
    mag = np.linalg.norm(grad, axis = -1)
    grad_dir = gradient_dir(img, r)
    threshold = grad_thresh * np.max(mag)
    thresholded = mag >= threshold
    xs, ys = np.indices(img.shape)
    xs_thresh = xs[thresholded]
    ys_thresh = ys[thresholded]
    lines_result = np.zeros(img.shape)
    for i in range(len(xs_thresh)):
        # set up convenient constants for quick reference
        orig_pos = (xs_thresh[i], ys_thresh[i])
        orig_x, orig_y = orig_pos
        orig_grad = grad[orig_pos]
        orig_grad_dir = grad_dir[orig_pos]
        grad_ortho = orthogonal_dirs(orig_grad_dir)[0]
        grad_ortho_x, grad_ortho_y = grad_ortho
        # figure out how long lines will be
        if grad_ortho_x == 0:
            x_len_p = img_max_x
            x_len_n = img_max_x
        elif grad_ortho_x > 0:
            x_len_p = int(((img_max_x - orig_x) / grad_ortho_x).round())
            x_len_n = int((orig_x / grad_ortho_x).round())
        else:
            x_len_p = int((orig_x / -grad_ortho_x).round())
            x_len_n = int(((img_max_x - orig_x) / -grad_ortho_x).round())
        if grad_ortho_y == 0:
            y_len_p = img_max_y
            y_len_n = img_max_y
        elif grad_ortho_y > 0:
            y_len_p = int(((img_max_y - orig_y) / grad_ortho_y).round())
            y_len_n = int((orig_y / grad_ortho_y).round())
        else:
            y_len_p = int((orig_y / -grad_ortho_y).round())
            y_len_n = int(((img_max_y - orig_y) / -grad_ortho_y).round())
        length_p = min([x_len_p, y_len_p, img_max_len])
        length_n = min([x_len_n, y_len_n, img_max_len])
        # spread information
        if length_p > 0:
            c = np.arange(1, length_p + 1)
            pos_x = orig_x + (c * grad_ortho_x).round().astype(int)
            pos_y = orig_y + (c * grad_ortho_y).round().astype(int)
            pos = np.unique(zip(pos_x, pos_y), axis = 0)
            pos_x = pos[:,0]
            pos_y = pos[:,1]
            grads = grad[zip(*pos)]
            amounts = np.inner(grads, orig_grad) # np.max([np.zeros(len(grads)), np.inner(grads, orig_grad)], axis = 0) for only positive dot products
            non_contiguous = amounts >= contig_thresh
            if any(non_contiguous):
                slicer = slice(0, non_contiguous.index(True))
                pos_x = pos_x[slicer]
                pos_y = pos_y[slicer]
                amounts = amounts[slicer]
            lines_result[pos_x, pos_y] += amounts
        if length_n > 0:
            c = np.arange(1, length_n + 1)
            pos_x = orig_x - (c * grad_ortho_x).round().astype(int)
            pos_y = orig_y - (c * grad_ortho_y).round().astype(int)
            pos = np.unique(zip(pos_x, pos_y), axis = 0)
            pos_x = pos[:,0]
            pos_y = pos[:,1]
            grads = grad[zip(*pos)]
            amounts = np.inner(grads, orig_grad) # np.max([np.zeros(len(grads)), np.inner(grads, orig_grad)], axis = 0) for only positive dot products
            non_contiguous = amounts >= contig_thresh
            if any(non_contiguous):
                slicer = slice(0, non_contiguous.index(True))
                pos_x = pos_x[slicer]
                pos_y = pos_y[slicer]
                amounts = amounts[slicer]
            lines_result[pos_x, pos_y] += amounts
#    normalizer = np.arange(curves_result.shape[2]).reshape((1,1,curves_result.shape[2],1))
#    curves_result[:,:,curve_r_range[0]:,:] /= normalizer[:,:,curve_r_range[0]:,:]
    return lines_result


### find features

# curves
def foci_connectivity(img, r = 1.5, coherence_r = 2.0, grad_thresh = 0.01, curve_r_range = None):#*** rewrite with new shifted_image ndarray capability
    focus = focus_pixels(img, r, grad_thresh, curve_r_range)[:,:,0,:]
    slicer = dists <= coherence_r
    dirs_r = dirs[slicer]
    floor_dists_r = floor_dists[slicer]
    int_dists_r = int_dists[slicer]
    ceil_dists_r = ceil_dists[slicer]
    are_int_dists_r = are_int_dists[slicer]
    foci = np.zeros(focus.shape + (len(dirs_r),))
    for d in range(len(dirs_r)):
        if are_int_dists_r[d]:
            dist = int_dists_r[d]
            # do subtraction as usual
            foci[:,:,0,d] = np.equal(focus[:,:,0] - dist, shifted_image(focus[:,:,0], dirs_r[d]))
            foci[:,:,1,d] = np.equal(focus[:,:,1] - dist, shifted_image(focus[:,:,1], dirs_r[d]))
        else:
            floor_dist = floor_dists_r[d]
            ceil_dist = ceil_dists_r[d]
            # do subtraction but also "or" in dist+1
            foci[:,:,0,d] = np.logical_or(np.equal(focus[:,:,0] - floor_dist, shifted_image(focus[:,:,0], dirs_r[d])), np.equal(focus[:,:,0] - ceil_dist, shifted_image(focus[:,:,0], dirs_r[d])))
            foci[:,:,1,d] = np.logical_or(np.equal(focus[:,:,1] - floor_dist, shifted_image(focus[:,:,1], dirs_r[d])), np.equal(focus[:,:,1] - ceil_dist, shifted_image(focus[:,:,1], dirs_r[d])))
    return foci

def foci(img, grad_r = 1.5, coherence_r = 2.0, grad_thresh = 0.01, curve_r_range = None):#*** rewrite with new shifted_image ndarray capability
    focus = focus_pixels(img, grad_r, grad_thresh, curve_r_range)
    focus[:,:,0,:] = np.zeros(img.shape + (2,))
    focus_1 = np.roll(focus, 1, axis = 2)
    focus_1[:,:,0,:] = np.zeros(img.shape + (2,))
    focus_2 = np.roll(focus_1, 1, axis = 2)
    focus_2[:,:,0,:] = np.zeros(img.shape + (2,))
    slicer = dists <= coherence_r
    dirs_r = dirs[slicer]
    int_dists_r = int_dists[slicer]
    are_int_dists_r = are_int_dists[slicer]
    foci = np.zeros(focus.shape + (len(dirs_r)+1,))
    for r in range(focus.shape[2]):
        foci[:,:,r,0,0] = focus[:,:,r,0]
        foci[:,:,r,1,0] = focus[:,:,r,1]
        for d in range(len(dirs_r)):
            if are_int_dists_r[d]:
                if int_dists_r[d] == 1:
                    foci[:,:,r,0,d+1] = shifted_image(focus_1[:,:,r,0], dirs_r[d])
                    foci[:,:,r,1,d+1] = shifted_image(focus_1[:,:,r,1], dirs_r[d])
                else:
                    foci[:,:,r,0,d+1] = shifted_image(focus_2[:,:,r,0], dirs_r[d])
                    foci[:,:,r,1,d+1] = shifted_image(focus_2[:,:,r,1], dirs_r[d])
            else:
                foci[:,:,r,0,d+1] = np.max([shifted_image(focus_1[:,:,r,0], dirs_r[d]), shifted_image(focus_2[:,:,r,0], dirs_r[d])], axis = 0)
                foci[:,:,r,1,d+1] = np.max([shifted_image(focus_1[:,:,r,1], dirs_r[d]), shifted_image(focus_2[:,:,r,1], dirs_r[d])], axis = 0)
    return foci
