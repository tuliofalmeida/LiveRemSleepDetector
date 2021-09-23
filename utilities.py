import numpy as np
from struct import Struct
import logging


frames_per_block = 128
size_timestamp = 4
size_sample = 2
size_magic_number = 4
waveform_bytes_per_frame = size_timestamp + size_sample
waveform_bytes_per_block = frames_per_block * waveform_bytes_per_frame + size_magic_number
logger = logging.getLogger('LRDlog')


def parse_block(raw_data, n_channels=1):
    # Improved version of Intan's code
    # we should time it
    expected_data_size = frames_per_block * (size_sample * n_channels + size_timestamp) + size_magic_number
    frame_struct = f'i{"H"*n_channels}' * frames_per_block
    block_struct = f'L{frame_struct}'
    # print(len(raw_data), expected_data_size, n_channels)
    if len(raw_data) % expected_data_size != 0:
        print('Data error')
        return
    if len(raw_data) == 0:
        return
    num_blocks = int(len(raw_data) / expected_data_size)
    all_block_struct = Struct(f'<{block_struct * num_blocks}')
    r = np.array(all_block_struct.unpack(raw_data))
    blocks = r.reshape((num_blocks, -1))
    if not np.all(blocks[:, 0] == 0x2ef07a08):
        logger.error(f'Error... magic number incorrect. Data size: {len(raw_data)}')
    data = blocks[:, 2::n_channels+1].reshape(-1)
    return data
