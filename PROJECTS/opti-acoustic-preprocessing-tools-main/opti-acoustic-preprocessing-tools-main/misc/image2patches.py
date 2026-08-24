#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Author: Hayat Rajani (hayat.rajani@udg.edu)
# Date: 2025-10-07
# Description: A utility for cutting an image into (overlapping) tiles
# of the specified dimensions.
# ---------------------------------------------------------------------


import cv2
import numpy


def createPatches(np_image, patch_shape, stride):

    if not isinstance(np_image, numpy.ndarray):
        raise ValueError("'np_image' must be a numpy ndarray!")

    if np_image.ndim < 2:
        raise ValueError("Cannot operate on 1D-arrays!")

    if len(patch_shape) < 2:
        raise ValueError("Incompatible dimensions! patch: %d, image: %d" %
                         (len(patch_shape), np_image.ndim))

    if len(stride) < 2:
        raise ValueError("Incompatible dimensions! stride: %d, image: %d" %
                         (len(stride), np_image.ndim))

    if any(a > b for a, b in zip(patch_shape, np_image.shape)):
        raise ValueError("Patch cannot be larger than image! patch {}, image {}".format(
            patch_shape, np_image.shape[:2]))

    if any(x <= 0 for x in patch_shape):
        raise ValueError(
            "Patch dimensions {} must be positive and non-zero!".format(patch_shape))

    if any(x <= 0 for x in stride):
        raise ValueError("Stride {} must be positive and non-zero!".format(stride))

    H, W = np_image.shape[:2]
    ph, pw = patch_shape
    sh, sw = stride

    # Adjust stride to evenly cover the whole image
    num_patches_y = int(numpy.ceil((H - ph) / sh)) + 1
    num_patches_x = int(numpy.ceil((W - pw) / sw)) + 1
    if num_patches_y > 1:
        sh = (H - ph) // (num_patches_y - 1)
    if num_patches_x > 1:
        sw = (W - pw) // (num_patches_x - 1)

    # Extract patches
    shape = (num_patches_y, num_patches_x, ph, pw)
    strides = (np_image.strides[0] * sh,
               np_image.strides[1] * sw,
               np_image.strides[0],
               np_image.strides[1])
    if np_image.ndim > 2:
        shape += (np_image.shape[2],)
        strides += (np_image.strides[2],)
    patches = numpy.lib.stride_tricks.as_strided(
        np_image, shape=shape, strides=strides)
    
    return patches


def read_image(file_path, allowed_extensions=('.png', '.PNG', '.jpg', '.JPG', '.tiff')):
    np_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) \
            if file_path.endswith(allowed_extensions) else \
        numpy.load(file_path) if file_path.endswith('.npy') else None
    if np_image is None:
        raise IOError('Unsupported file format!')
    return np_image


def save_patches(patches, out_dir, out_file):
    out_file, ext = os.path.basename(out_file).rsplit('.',1)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            out_path = os.path.join(out_dir, f'{out_file}_r{i}_c{j}.{ext}')
            cv2.imwrite(out_path, patches[i,j]) if ext != 'npy' else \
                numpy.save(out_path, patches[i,j], allow_pickle=False)


if __name__ == '__main__':

    import os
    import sys
    from datetime import datetime

    try:
        data_dir = sys.argv[1]
        patch_shape = (int(sys.argv[2]), int(sys.argv[3]))
        stride = (int(sys.argv[4]), int(sys.argv[5]))
    except:
        print('Usage: python3 image2patches.py <dataset_dir>',
              '<patch_height> <patch_width> <row_stride> <col_stride>')
    else:
        if os.path.exists(data_dir):
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            patch_dir = 'patches_' + timestamp

            if os.path.isfile(data_dir):
                dir_name, file_name = os.path.split(data_dir)
                os.chdir(dir_name)
                try:
                    np_image = read_image(file_name)
                    patches = createPatches(np_image, patch_shape, stride)
                    save_patches(patches, patch_dir, file_name)
                except (IOError, ValueError) as e:
                    print(e)
                else:
                    print('Patches saved!')

            if os.path.isdir(data_dir):
                os.chdir(data_dir)
                for file_name in os.listdir():
                    if os.path.isfile(file_name):
                        out_dir = os.path.join(patch_dir, file_name.split('.')[0])
                        try:
                            print('\nProcessing', file_name, '...')
                            np_image = read_image(file_name)
                            patches = createPatches(np_image, patch_shape, stride)
                            save_patches(patches, out_dir, file_name)
                        except (IOError, ValueError) as e:
                            print(e)
                        else:
                            print('\tPatches saved!')
                print('\nDone!')
        else:
            sys.exit('Invalid Path!')
