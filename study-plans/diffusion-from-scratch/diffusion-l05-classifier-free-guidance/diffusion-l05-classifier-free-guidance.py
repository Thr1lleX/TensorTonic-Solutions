import torch

def classifier_free_guidance(cond_pred: torch.Tensor, uncond_pred: torch.Tensor, guidance_scale: float, guidance_rescale: float) -> torch.Tensor:
    """
    Returns the guided prediction with the shape and dtype of cond_pred.
    """
    
    g = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
    dims = tuple(range(1, g.ndim))
    rescaled_prediction = g * cond_pred.std(dim = dims, keepdim = True, correction = 0)/g.std(dim = dims, keepdim = True, correction = 0)

    result = guidance_rescale * rescaled_prediction + (1-guidance_rescale)*g
    return result
