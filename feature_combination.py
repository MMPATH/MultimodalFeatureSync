import pandas as pd
from utilities import round_to_base, resample_features


def combine_features(
    facial_features, vocal_features,
    linguistic_features, alignment_base=0.01
):
    """
    Combines facial, vocal, and linguistic features into a single DataFrame.

    Args:
    facial_features (DataFrame): Facial features.
    vocal_features (DataFrame): Vocal features.
    linguistic_features (DataFrame): Linguistic features.
    alignment_base (float): Base for rounding time indices for alignment.

    Returns:
    DataFrame: Combined features.
    """
    # Align indices
    facial_features.index = facial_features.index.map(
        lambda x: round_to_base(x, alignment_base)
    )
    vocal_features.index = vocal_features.index.map(
        lambda x: round_to_base(x, alignment_base)
    )
    linguistic_features.index = linguistic_features.index.map(
        lambda x: round_to_base(x, alignment_base)
    )

    # Resample features
    resampled_facial = resample_features(facial_features, vocal_features.index)
    resampled_linguistic = resample_features(linguistic_features,
                                             vocal_features.index)

    # Combine features
    combined = pd.concat(
        [resampled_facial, vocal_features, resampled_linguistic], axis=1
    )
    return combined.dropna()
