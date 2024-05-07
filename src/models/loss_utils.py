import torch


# rotation matrix based angle loss
def rotation_loss(angs_a, angs_b):
    """computes the angular error between two lists of angles in radians

    loss: |I - R_a^T @ R_b|
    """

    def rotmat_from_angle(z):
        rotmat = torch.stack([
            torch.stack([torch.cos(z), -torch.sin(z)]),
            torch.stack([torch.sin(z), torch.cos(z)])
        ])

        return rotmat.permute(2, 0, 1)

    # R_a^T @ R_b
    rotmat_mult = rotmat_from_angle(angs_a).permute(0, 2, 1) @ rotmat_from_angle(angs_b)

    loss_rot = (torch.eye(2).to(rotmat_mult.device) - rotmat_mult).norm('fro', -1).norm('fro', -1).mean()

    return loss_rot

