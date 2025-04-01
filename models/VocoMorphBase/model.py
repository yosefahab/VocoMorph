import torch
import torch.nn as nn

from .modules.tcn import TCN
from .modules.condition_encoder import ConditionEncoder


class VocoMorphBase(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.tcn = TCN(config["module_tcn"])
        self.condition_encoder = ConditionEncoder(config["module_condition_encoder"])
        self.input_dim = self.tcn.input_dim
        self.overlap = 0

    def forward(self, x):
        """
        Process audio in sequential chunks

        Args:
            x: Input audio tensor (B, C, L)
            effect_id: Effect ID tensor (B, 1)
            chunk_size: Size of each chunk to process
            overlap: Overlap between consecutive chunks
        """
        effect_id, x = x
        gamma, beta = self.condition_encoder(effect_id)

        # get dimensions
        _, _, length = x.shape

        # Initialize output tensor
        output = torch.zeros_like(x)

        # calculate effective history needed based on TCN receptive field
        # this depends on your kernel size and max dilation
        receptive_field = self.tcn.calculate_receptive_field()

        # process in chunks with overlap
        for i in range(0, length, self.input_dim - self.overlap):
            # calculate start and end positions
            start_idx = max(0, i - receptive_field)
            end_idx = min(length, i + self.input_dim)

            # extract chunk
            chunk = x[:, :, start_idx:end_idx]

            # apply FiLM conditioning
            chunk = gamma * chunk + beta

            # process through TCN
            processed_chunk = self.tcn(chunk)

            # calculate output region (discard receptive field padding if present)
            out_start = 0 if i == 0 else receptive_field
            out_end = processed_chunk.shape[2]

            # calculate where to put the output
            out_pos_start = start_idx + out_start
            out_pos_end = min(length, out_pos_start + (out_end - out_start))

            # add to output (or use crossfade for overlap regions)
            if i == 0 or self.overlap == 0:
                output[:, :, out_pos_start:out_pos_end] = processed_chunk[
                    :, :, out_start : out_start + (out_pos_end - out_pos_start)
                ]
            else:
                # TODO: implement crossfade logic here for overlapping regions
                pass

        return output
