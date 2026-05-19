import torch

def semantic_accuracy(pred_labels, true_labels):
    """Compute semantic accuracy: percentage of nodes with correct predicted label."""
    pred_labels = torch.argmax(pred_labels, dim=1) # (N,)
    true_labels = torch.argmax(true_labels, dim=1) # (N,)
    return (pred_labels == true_labels).float().mean().item()

def mean_position_error(pred_positions, true_positions):
    """
    The result is in the same units as the input positions.
    Since positions are normalised to [-1, 1], the error is also in that range.
    An error of 0.05 means 5 % of the scene's extent.
    """
    dist = torch.norm(pred_positions - true_positions, dim=1) # (N,) one distance per node
    return dist.mean().item() # average distance across all nodes


def mean_size_error(pred_sizes, true_sizes):
    dist = torch.norm(pred_sizes - true_sizes, dim=1) # (N,) one distance per node
    return dist.mean().item() # average distance across all nodes

def compute_metrics(outputs, stage):
    if stage < 2:
        return {}
    
    with torch.no_grad():
        metrics = {}

        # mid-level metrics
        metrics['acc_mid'] = semantic_accuracy(outputs['recon_mid']['s'], outputs['s_lm1'])
        metrics['pos_err_mid'] = mean_position_error(outputs['recon_mid']['p'], outputs['p_lm1'])
        metrics['size_err_mid'] = mean_size_error(outputs['recon_mid']['r'], outputs['r_lm1'])

        # coarse-level metrics
        metrics['acc_coarse'] = semantic_accuracy(outputs['recon_coarse']['s'], outputs['s_1'])
        metrics['pos_err_coarse'] = mean_position_error(outputs['recon_coarse']['p'], outputs['p_1'])
        metrics['size_err_coarse'] = mean_size_error(outputs['recon_coarse']['r'], outputs['r_1'])

    return metrics

