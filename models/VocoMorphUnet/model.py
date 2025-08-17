import mlx.core as mx
import mlx.nn as nn

from .modules.effect_encoder import EffectEncoder
from .modules.encoder import Encoder
from .modules.decoder import Decoder


class VocoMorphUnet(nn.Module):
    """
    The main VocoMorph U-Net model.

    It uses a U-Net architecture with FiLM layers to condition audio processing
    on a specified effect embedding.
    """

    def __init__(self, config: dict):
        super().__init__()
        # Retrieve config values.
        self.chunk_size = config["chunk_size"]
        self.num_channels = config["num_channels"]
        embedding_dim = config["embedding_dim"]
        encoder_filters = config["encoder_filters"]
        bottleneck_filters = config["bottleneck_filters"]
        decoder_filters = config["decoder_filters"]
        kernel_size = config["kernel_size"]
        padding = config["padding"]

        # New assertion to provide a more informative error message.
        assert self.num_channels in [1, 2], (
            f"VocoMorphUnet expects num_channels to be 1 (mono) or 2 (stereo), "
            f"but got {self.num_channels}. Please update your config file accordingly."
        )

        # Ensure encoder and decoder have a matching number of levels for skip connections.
        assert len(encoder_filters) == len(decoder_filters), (
            "Number of Encoder and Decoder blocks must match"
        )

        self.effect_encoder = EffectEncoder(config["num_effects"], embedding_dim)

        self.encoder_blocks = []
        for i in range(len(encoder_filters)):
            in_channels = self.num_channels if i == 0 else encoder_filters[i - 1]
            out_channels = encoder_filters[i]
            self.encoder_blocks.append(
                Encoder(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding,
                    embedding_dim,
                    apply_pool=True,
                )
            )

        self.bottleneck_blocks = []
        for i in range(len(bottleneck_filters)):
            in_channels = encoder_filters[-1] if i == 0 else bottleneck_filters[i - 1]
            out_channels = bottleneck_filters[i]
            self.bottleneck_blocks.append(
                Encoder(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding,
                    embedding_dim,
                    apply_pool=False,
                )
            )

        self.decoder_blocks = []
        num_decoders = len(decoder_filters)
        for i in range(num_decoders):
            # The previous channels are from the bottleneck or the previous decoder.
            prev_channels = bottleneck_filters[-1] if i == 0 else decoder_filters[i - 1]
            # The skip channels are from the corresponding encoder, so they are reversed.
            skip_channels = encoder_filters[num_decoders - 1 - i]

            self.decoder_blocks.append(
                Decoder(
                    prev_channels,
                    skip_channels,
                    decoder_filters[i],
                    kernel_size,
                    padding,
                )
            )

        # The final convolution layer reduces the channels back to the original `num_channels`.
        self.final_conv = nn.Conv1d(
            decoder_filters[-1], self.num_channels, kernel_size=1
        )
        self.num_downsampling_steps = len(encoder_filters)

    def __call__(self, effect_id: mx.array, audio_chunk: mx.array):
        # We assume the input `audio_chunk` is in the format (Batch, Channels, Length).
        # We must transpose it to (Batch, Length, Channels) to match MLX's conv1d expectation.
        x = audio_chunk.transpose(0, 2, 1)

        # Get the conditioning embedding from the effect_id.
        effect_embedding = self.effect_encoder(effect_id)

        skip_connections = []

        # Encoder path.
        for encoder in self.encoder_blocks:
            x, skip = encoder(x, effect_embedding)
            skip_connections.append(skip)

        # Bottleneck path.
        for bottleneck in self.bottleneck_blocks:
            x, _ = bottleneck(x, effect_embedding)

        # Decoder path.
        # We process the decoder blocks and their corresponding skip connections in reverse.
        for decoder, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder(x, skip)

        # Final convolution to map channels back to the original count.
        x = self.final_conv(x)

        # Transpose the output back to the original shape (Batch, Channels, Length)
        # to ensure the final output format is what the user expects.
        x = x.transpose(0, 2, 1)

        # Final shape assertion to confirm everything worked as expected.
        assert x.shape == audio_chunk.shape, (
            f"Model output shape ({x.shape}) does not match input shape ({audio_chunk.shape})"
        )
        return x


if __name__ == "__main__":
    # dummy config for a simple test
    config = {
        "num_channels": 1,
        "chunk_size": 16384,
        "embedding_dim": 64,
        "num_effects": 10,
        "encoder_filters": [32, 64, 128],
        "bottleneck_filters": [256],
        "decoder_filters": [128, 64, 32],
        "kernel_size": 3,
        "padding": 1,
    }

    # initialize the model with the dummy config
    print("initializing VocoMorphUnet model...")
    model = VocoMorphUnet(config)
    print("model initialized successfully!")

    # create dummy input data
    batch_size = 4
    audio_chunk = mx.random.normal(
        shape=(batch_size, config["num_channels"], config["chunk_size"])
    )
    effect_id = mx.array([0, 1, 2, 3])

    # perform a dummy forward pass and print the output shape
    print("\nperforming a dummy forward pass...")
    output = model(effect_id, audio_chunk)

    # print the output shape to verify it's correct
    print(f"input audio chunk shape: {audio_chunk.shape}")
    print(f"output audio chunk shape: {output.shape}")
    print("\nmodel test completed successfully!")
