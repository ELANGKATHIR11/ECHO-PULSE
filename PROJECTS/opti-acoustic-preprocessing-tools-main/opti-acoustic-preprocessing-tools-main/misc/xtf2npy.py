#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Author: Hayat Rajani (hayat.rajani@udg.edu)
# Date: 2025-10-07
# Description: A utility for extracting waterfalls from XTF file(s) as
# NPY arrays or TIFF images. Note that NPY arrays preserve bit depth,
# whereas TIFF images quantize the waterfalls to 8 bits per pixel.
# ---------------------------------------------------------------------


import os
import numpy
import imageio

from math import ceil


def blindZoneIdx(sonar_pings):
    n_samples = sonar_pings[0].ping_chan_headers[0].NumSamples * 2
    slant_res = sonar_pings[0].ping_chan_headers[0].SlantRange * 2 / n_samples

    h = numpy.max([ping.SensorPrimaryAltitude for ping in sonar_pings])
    
    n_bins_blind = int(round(h / slant_res))            # per side
    n_bins_ground = int(n_samples / 2 - n_bins_blind)   # per side

    port_idx = slice(24, n_bins_ground)
    stbd_idx = slice(n_bins_blind, -24)

    #port_idx = slice(None, n_bins_ground)
    #stbd_idx = slice(n_bins_blind, None)

    return port_idx, stbd_idx


def createWaterfall(sonar_pings, bit_depth, logarithmic=False, 
        batch_size=None, split=False, unblind=False, **kwargs):

    file_name = kwargs.get('file_name', None)
    out_dir = kwargs.get('out_dir', None)
    out_file = kwargs.get('out_file', None)
    out_format = kwargs.get('out_format', 'npy')

    bit_depth = 2 ** bit_depth - 1

    if sonar_pings is None:
        if file_name is None:
            raise TypeError("Cannot operate on None type! At least one argument required, \
                            'sonar_image' or 'file_name'")
        _, packet = pyxtf.xtf_read(file_name)
        sonar_pings = packet[pyxtf.XTFHeaderType.sonar]
    n_pings = len(sonar_pings)
    n_channels = len(sonar_pings[0].data)

    if not batch_size:
        batch_size = n_pings
 
    sonar_images = [[None for i in range(n_channels)]
        for n in range(ceil(n_pings/batch_size))]
    
    for n in range(0, n_pings, batch_size):
        idx = n // batch_size
        ping_idx = slice(n, n + batch_size)
        for i in range(n_channels):
            # Concatenate all individual pings for a channel in a dense array
            sonar_images[idx][i] = numpy.vstack([
                ping.data[i] for ping in sonar_pings[ping_idx]
            ])
            sonar_images[idx][i] = numpy.log1p(sonar_images[idx][i])/numpy.log1p(bit_depth) \
                if logarithmic else sonar_images[idx][i]/bit_depth

        if unblind:
            # Get indices of bins outside the blind zone
            port_idx, stbd_idx = blindZoneIdx(sonar_pings[ping_idx])
            for i in range(0, n_channels, 2):
                # Remove blind zone
                sonar_images[idx][i] = sonar_images[idx][i][:, port_idx]
                sonar_images[idx][i + 1] = sonar_images[idx][i + 1][:, stbd_idx]

        if not split:
            # Join port and starboard channels into a single waterfall
            sonar_images[idx] = [numpy.hstack((
                sonar_images[idx][i], sonar_images[idx][i + 1]
            )) for i in range(0, n_channels, 2)]

    if not out_dir is None:
        os.makedirs(out_dir, exist_ok=True)
        if out_file is None:
            out_file = os.path.basename(file_name).split('.')[0] \
                if not file_name is None and os.path.isfile(file_name) else 'waterfall'
        for n, sonar_image in enumerate(sonar_images):
            for ch, sonar_channel in enumerate(sonar_image):
                out_path = os.path.join(out_dir, f'{out_file}_batch{n}_ch{ch}.{out_format}')
                if out_format == 'npy':
                    numpy.save(out_path, sonar_channel, allow_pickle=False)
                else:
                    imageio.imwrite(out_path, (sonar_channel*255).round().astype(numpy.uint8))
    else:
        return sonar_images[0] if batch_size == n_pings else sonar_images


if __name__ == '__main__':

    import sys
    import argparse
    import shutil
    import pyxtf

    from datetime import datetime

    argparser = argparse.ArgumentParser(description=
        'Extracts waterfalls from XTF file(s) as NPY arrays or TIFF images. ' \
        'Note that NPY arrays preserve the bit depth, whereas TIFF images quantize the waterfalls to 8 bits per pixel.')
    argparser.add_argument('--path', required=True,
        help='path to XTF file(s)')
    argparser.add_argument('--bits', choices=[8,12,16], type=int, required=True,
        help='input bit depth [8|12|16]')
    argparser.add_argument('--batch', metavar='N', type=int, default=1152,
        help="save waterfalls in batches of 'N' pings (default: 1152)")
    argparser.add_argument('--split', action='store_true',
        help='split port and starboard waterfalls')
    argparser.add_argument('--unblind', action='store_true',
        help='remove the blind zone / water column')
    argparser.add_argument('--log', action='store_true',
        help='use logarithmic intensities')
    argparser.add_argument('--format', choices=['tiff', 'npy'], default='npy',
        help='output format [tiff|npy] (default: npy)')
    args = argparser.parse_args()

    if os.path.exists(args.path):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        out_dir = f'out_{timestamp}'

        if os.path.isfile(args.path):
            dir_name, file_name = os.path.split(args.path)
            os.chdir(dir_name)
            try:
                createWaterfall(None, args.bits, args.log, args.batch, args.split, args.unblind,
                    out_format=args.format, file_name=file_name, out_dir=out_dir)
            except ValueError as e:
                print(e)
            else:
                print('Waterfall saved!')

        if os.path.isdir(args.path):
            os.chdir(args.path)
            err_dir = f'errors_{timestamp}'
            os.mkdir(err_dir)
            log_file = os.path.join(err_dir, 'log.txt')

            for file_name in os.listdir():
                if file_name.endswith('.xtf'):
                    print('\nProcessing', file_name, '...')
                    try:
                        createWaterfall(None, args.bits, args.log, args.batch,
                            args.split, args.unblind, out_format=args.format,
                            file_name=file_name, out_dir=out_dir)
                    except Exception as e:
                        msg = 'Dimension mismatch: swath widths not uniform!' \
                            if e.__class__ is ValueError else \
                                'Tile out of image; possibly due to max altitude!' \
                                    if e.__class__ is SystemError else str(e)
                        print(msg, 'Skipping...\n')
                        shutil.copy(file_name, err_dir + os.sep + file_name)
                        with open(log_file, 'w') as f:
                            print(file_name, ': ', msg, '\n', file=f)
                        continue
                    else:
                        print('\tWaterfall saved!')

            print('\nDone!')
    else:
        sys.exit('Invalid Path!')
