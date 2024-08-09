"""Add functions for postprocessing."""

def one_stage_post_processing(
        pred_volume_fraction_1: float, pred_volume_fraction_2: float,
) -> list[float, float]:
    """Apply one-stage-post-processing."""
    total_pred_volume_fractions = pred_volume_fraction_1 + pred_volume_fraction_2
    postprocessed_volume_fraction_1 = pred_volume_fraction_1 / total_pred_volume_fractions
    postprocessed_volume_fraction_2 = pred_volume_fraction_2 / total_pred_volume_fractions
    return [postprocessed_volume_fraction_1, postprocessed_volume_fraction_2]


def two_stage_post_processing(
        pred_volume_fraction_1: float, pred_volume_fraction_2: float,
) -> list[float, float]:
    """Apply two-stage-post-processing."""
    new_pred_volume_fraction_1 = (pred_volume_fraction_1 + 1 - pred_volume_fraction_2) / 2
    new_pred_volume_fraction_2 = (pred_volume_fraction_2 + 1 - pred_volume_fraction_1) / 2
    total_pred_volume_fractions = new_pred_volume_fraction_1 + new_pred_volume_fraction_2
    postprocessed_volume_fraction_1 = new_pred_volume_fraction_1 / total_pred_volume_fractions
    postprocessed_volume_fraction_2 = new_pred_volume_fraction_2 / total_pred_volume_fractions
    return [postprocessed_volume_fraction_1, postprocessed_volume_fraction_2]
