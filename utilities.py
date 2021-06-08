import numpy as np
from struct import Struct


frames_per_block = 128
waveform_bytes_per_frame = 4 + 2
waveform_bytes_per_block = frames_per_block * waveform_bytes_per_frame + 4
frame_struct = 'iH' * frames_per_block
block_struct = f'L{frame_struct}'


def parse_block(raw_data):
    # Improved version of Intan's code
    # we should time it
    if len(raw_data) % waveform_bytes_per_block != 0:
        print('Data error')
        return
    if len(raw_data) == 0:
        return
    num_blocks = int(len(raw_data) / waveform_bytes_per_block)
    all_block_struct = Struct(f'<{block_struct * num_blocks}')
    r = np.array(all_block_struct.unpack(raw_data))
    blocks = r.reshape((num_blocks, -1))
    if not np.all(blocks[:, 0] == 0x2ef07a08):
        raise ValueError(f'Error... magic number {blocks[0, 0]} incorrect')
    data = blocks[:, 2::2].reshape(-1)
    return data
