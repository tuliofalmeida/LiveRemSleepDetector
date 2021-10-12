import numpy as np
from struct import Struct
import logging
from functools import lru_cache


frames_per_block = 128
size_timestamp = 4
size_sample = 2
size_magic_number = 4
waveform_bytes_per_frame = size_timestamp + size_sample
waveform_bytes_per_block = frames_per_block * waveform_bytes_per_frame + size_magic_number
logger = logging.getLogger('LRDlog')


@lru_cache
def get_data_struct(data_size, n_channels=1):
    expected_data_size = frames_per_block * (size_sample * n_channels + size_timestamp) + size_magic_number
    frame_struct = f'i{"H"*n_channels}' * frames_per_block
    block_struct = f'L{frame_struct}'
    # print(len(raw_data), expected_data_size, n_channels)
    if data_size % expected_data_size != 0:
        print('Data error')
        return None, None
    if data_size == 0:
        return None, None
    num_blocks = int(data_size / expected_data_size)
    all_block_struct = Struct(f'<{block_struct * num_blocks}').unpack
    return num_blocks, all_block_struct


def parse_block(raw_data, n_channels=1):
    # Improved version of Intan's code
    # we should time it
    num_blocks, all_block_struct = get_data_struct(len(raw_data), n_channels)
    if all_block_struct is None:
        return all_block_struct
    r = np.array(all_block_struct(raw_data))
    blocks = r.reshape((num_blocks, -1))
    if not np.all(blocks[:, 0] == 0x2ef07a08):
        logger.error(f'Error... magic number incorrect. Data size: {len(raw_data)}')
    samples = np.vstack([blocks[:, ix + 2::n_channels+1].reshape(-1) for ix in range(n_channels)])
    timestamps = blocks[:, 1::n_channels+1].reshape(-1)
    data = np.vstack((timestamps, samples))
    return data
