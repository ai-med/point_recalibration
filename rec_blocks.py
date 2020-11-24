"""
Implementation of the re-calibration blocks from the paper
"Recalibration of Neural Networks for Point Cloud Analysis "
Presented at 3DV 2020.
Authors: Ignacio Sarasua, Sebastian Poelsterl and Christian Wachinger
"""


import torch
import torch.nn as nn



class PointCRB(nn.Module):
    """
        Channel Re-calibration block.
        Init:
            in_channel: number of channels of the input feature map
            r: re-calibration factor (default:2)
        """
    def __init__(self, in_channel, r = 2):
        super(PointCRB, self).__init__()
        num_channels_reduced = in_channel // r
        self.reduction_ratio = in_channel
        self.fc1 = nn.Linear(in_channel, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, in_channel, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, points):
        """
        Input:
            points: feature map
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        mask = torch.mean(points,2)
        mask = self.relu(self.fc1(mask))
        mask = torch.sigmoid(self.fc2(mask))
        a, b = mask.size()
        out_points = torch.mul(points, mask.view(a, b, 1))
        return out_points


class PointSRB(nn.Module):
    """
    Spatial Re-calibration block.
    Init:
        in_channel: number of channels of the input feature map
        n_points: number of points of the input feature map
        r: re-calibration factor (default:2)


    """
    def __init__(self, in_channel,n_points,r=2):
        super(PointSRB, self).__init__()
        self.conv = nn.Conv1d(in_channel,1,1)
        self.n_points = n_points
        n_points_reduced = int(n_points/r)
        self.fc1 = nn.Linear(n_points, n_points_reduced, bias=True)
        self.fc2 = nn.Linear(n_points_reduced, n_points, bias=True)
        self.relu = nn.ReLU()

    def forward(self, points):
        """
        Input:
            points: feature map
        Return:
            out_points: re-calibrated feature map
        """
        B,C,N = points.size()
        mask = self.conv(points)
        mask = mask.transpose(2,1)
        mask = self.relu(self.fc1(mask))
        mask = torch.sigmoid(self.fc2(mask))
        mask = mask.view(B, 1,N)
        out_points = torch.mul(points, mask)
        return out_points


class PointSCRB(nn.Module):
    """
    Spatial-Channel Re-calibration block.
    Init:
        in_channel: number of channels of the input feature map
        n_points: number of points of the input feature map
        r: re-calibration factor (default:2)

    """

    def __init__(self, in_channel,n_points,r=2):
        super(PointSCRB, self).__init__()
        self.cSE = PointCRB(in_channel, r)
        self.sSE = PointSRB(in_channel, n_points,r)

    def forward(self, points):
        """
        Input:
            points: feature map
        Return:
            out_points: re-calibrated feature map
        """
        output_points = torch.max(self.cSE(points), self.sSE(points))
        return output_points
