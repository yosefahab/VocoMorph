import torch
import torch.nn as nn
from .modules.tcn import TCN
from .modules.condition_encoder import ConditionEncoder


class VocoMorphBase(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.chunk_size = config["chunk_size"]
        self.overlap = config["overlap"]

        self.tcn = TCN(config["module_tcn"])
        self.condition_encoder = ConditionEncoder(config["module_condition_encoder"])

    def forward(self, x):
        effect_id, audio = x
        embedding = self.condition_encoder(effect_id)

        _, _, T = audio.shape
        output = torch.zeros_like(audio)
        receptive_field = self.tcn.calculate_receptive_field()

        for i in range(0, T, self.chunk_size - self.overlap):
            start_idx = max(0, i - receptive_field)
            end_idx = min(T, i + self.chunk_size)
            chunk = audio[:, :, start_idx:end_idx]

            processed_chunk = self.tcn(chunk, embedding)

            out_start = 0 if i == 0 else receptive_field
            out_end = processed_chunk.shape[2]
            out_pos_start = start_idx + out_start
            out_pos_end = min(T, out_pos_start + (out_end - out_start))

            if i == 0 or self.overlap == 0:
                output[:, :, out_pos_start:out_pos_end] = processed_chunk[
                    :, :, out_start : out_start + (out_pos_end - out_pos_start)
                ]
            else:
                # TODO: implement crossfade logic for overlapping regions
                pass

        return output
