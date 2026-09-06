from numpy import int16
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is None:
        return values.mean()

    mask = mask.to(dtype=values.dtype, device=values.device).reshape(-1)
    values = values.reshape(-1)
    denom = mask.sum()
    if denom.item() <= 0:
        return values.new_zeros(())
    return (values * mask).sum() / denom

 
class DockQPairwiseThresholdLoss(nn.Module):
    def __init__(
        self,
        lambda_pair=1.0,
        huber_delta=0.1,
    ):
        super().__init__()
        self.lambda_pair = lambda_pair
        self.huber_delta = huber_delta

        self.reg_criterion = nn.MSELoss("mean")
        self.pair_criterion = nn.HuberLoss(delta=self.huber_delta, reduction='mean')


    def forward(
        self,
        y1_pred, y2_pred,
        y1_true, y2_true,
        pair_mask=None,
    ):
        """
        y*_pred, y*_true: shape (B,)
        """

        # ========= Absolute regression =========
        loss_reg_1 = self.reg_criterion(y1_pred, y1_true)
        loss_reg_2 = masked_mean((y2_pred - y2_true) ** 2, pair_mask)
        loss_reg = loss_reg_1 + loss_reg_2

        # ========= Pairwise relative loss =========
        pred_diff = y2_pred - y1_pred
        true_diff = y2_true - y1_true

        pair_abs = F.smooth_l1_loss(pred_diff, true_diff, beta=self.huber_delta, reduction='none')
        loss_pair = masked_mean(pair_abs, pair_mask)

        # ========= Total loss =========
        loss = (loss_reg + self.lambda_pair * loss_pair)
        return loss, loss_reg.detach(), loss_pair.detach()

    def cal_test_loss(self, y_pred, y_true):
        loss_reg = F.mse_loss(y_pred, y_true)
        return loss_reg



class WeightedDockQPairwiseThresholdLoss(nn.Module):
    def __init__(
        self,
        lambda_pair=0.5,
        huber_delta=0.1,
        thr=0.49,
        # sample weighting
        w_mid=4.0,
        w_cross=3.0,
        mid_lo=0.4,
        mid_hi=0.6,
        # pair filtering
        min_true_gap=0.05,
        use_huber_reg=True,
    ):
        super().__init__()
        self.lambda_pair = lambda_pair
        self.huber_delta = huber_delta
        self.thr = thr

        self.w_mid = w_mid
        self.w_cross = w_cross
        self.mid_lo = mid_lo
        self.mid_hi = mid_hi

        self.min_true_gap = min_true_gap
        self.use_huber_reg = use_huber_reg

    def _sample_weight(self, y_true: torch.Tensor) -> torch.Tensor:
        """Per-sample weight emphasizing threshold band."""
        w = torch.ones_like(y_true)
        mid = (y_true >= self.mid_lo) & (y_true <= self.mid_hi)
        w[mid] = self.w_mid
        return w

    def _weighted_reg_loss(self, y_pred, y_true, sample_mask=None):
        w = self._sample_weight(y_true)
        if sample_mask is not None:
            w = w * sample_mask.to(dtype=w.dtype, device=w.device).reshape_as(w)

        if self.use_huber_reg:
            # huber (smooth_l1) per-sample
            per = F.smooth_l1_loss(y_pred, y_true, beta=self.huber_delta, reduction="none")
        else:
            per = (y_pred - y_true) ** 2  # mse per-sample

        loss = (w * per).sum() / (w.sum() + 1e-8)
        return loss

    def forward(self, y1_pred, y2_pred, y1_true, y2_true, pair_mask=None):
        """
        y*_pred, y*_true: shape (B,)
        """

        # ========= Absolute regression (weighted) =========
        if pair_mask is None:
            pair_mask = torch.ones_like(y1_true, dtype=y1_true.dtype, device=y1_true.device)

        loss_reg = self._weighted_reg_loss(y1_pred, y1_true) + self._weighted_reg_loss(y2_pred, y2_true, sample_mask=pair_mask)

        # ========= Pairwise ranking loss (weighted) =========
        pred_diff = y2_pred - y1_pred
        true_diff = y2_true - y1_true

        # ignore near-ties: too noisy, especially around threshold
        mask = (true_diff.abs() >= self.min_true_gap).float()
        mask = mask * pair_mask.to(dtype=mask.dtype, device=mask.device).reshape_as(mask)

        # sign: want sign*pred_diff large positive
        sign = torch.sign(true_diff)  # +1 / -1 / 0

        # logistic ranking loss: log(1 + exp(- sign * pred_diff))
        base = torch.log1p(torch.exp(-sign * pred_diff))

        # pair weights: emphasize crossing threshold & threshold-band pairs
        w_pair = torch.ones_like(true_diff)

        cross = ((y1_true - self.thr) * (y2_true - self.thr) < 0)  # one below, one above
        w_pair[cross] *= self.w_cross

        mid_pair = (
            (y1_true >= self.mid_lo) & (y1_true <= self.mid_hi) &
            (y2_true >= self.mid_lo) & (y2_true <= self.mid_hi)
        )
        w_pair[mid_pair] *= self.w_mid

        denom = (w_pair * mask).sum() + 1e-8
        loss_pair = ((w_pair * mask) * base).sum() / denom

        # ========= Total loss =========
        loss = loss_reg + self.lambda_pair * loss_pair

        return loss, {
            "reg": loss_reg.detach(),
            "pair": loss_pair.detach(),
            "pair_denom": denom.detach(),
        }

    def cal_test_loss(self, y_pred, y_true):
        # test: report plain mse + (optional) weighted mse
        loss_mse = F.mse_loss(y_pred, y_true)
        loss_wreg = self._weighted_reg_loss(y_pred, y_true)
        return loss_wreg, {
            "mse": loss_mse.detach(),
            "wreg": loss_wreg.detach(),
        }

class ThresholdDockQPairwiseThresholdLoss(nn.Module):
    def __init__(
        self,
        tau=0.49,
        margin=0.05,
        lambda_pair=1.0,
        lambda_thr=0.2,
        sigma=0.08, 
        use_region_mask=True,
        huber_delta=0.1,
    ):
        super().__init__()
        self.tau = tau
        self.margin = margin
        self.lambda_pair = lambda_pair
        self.lambda_thr = lambda_thr
        self.sigma = sigma
        self.use_region_mask = use_region_mask
        self.huber_delta = huber_delta

    def forward(
        self,
        y1_pred, y2_pred,
        y1_true, y2_true,
        pair_mask=None,
    ):
        """
        y*_pred, y*_true: shape (B,)
        """

        # ========= Absolute regression =========
        loss_reg_1 = F.mse_loss(y1_pred, y1_true)
        loss_reg_2 = masked_mean((y2_pred - y2_true) ** 2, pair_mask)
        loss_reg = loss_reg_1 + loss_reg_2

        # ========= Pairwise relative loss =========
        pred_diff = y2_pred - y1_pred
        true_diff = y2_true - y1_true

        pair_abs = F.smooth_l1_loss(
            pred_diff,
            true_diff,
            beta=self.huber_delta,
            reduction="none",
        )
        loss_pair = masked_mean(pair_abs, pair_mask)

        # ========= Threshold-aware margin loss =========
        loss_thr = 0.5 * (
            self.threshold_margin(y1_pred, y1_true)
            + self.threshold_margin(y2_pred, y2_true, sample_mask=pair_mask)
        )

        # ========= Total loss =========
        loss = (
            loss_reg
            + self.lambda_pair * loss_pair
            + self.lambda_thr * loss_thr
        )

        return loss, {
            "reg": loss_reg.detach(),
            "pair": loss_pair.detach(),
            "thr": loss_thr.detach(),
        }
    

    def cal_test_loss(self, y_pred, y_true):
        loss_reg = F.mse_loss(y_pred, y_true)
        loss_thr = self.threshold_margin(y_pred, y_true)
        loss = loss_reg + self.lambda_thr * loss_thr
        return loss, {
            "reg": loss_reg.detach(),
            "thr": loss_thr.detach(),
        }


    def threshold_margin(self, y_pred, y_true, sample_mask=None):
        t = torch.where(
            y_true >= self.tau,
            torch.ones_like(y_true),
            -torch.ones_like(y_true),
        )

        signed_dist = t * (y_pred - self.tau)
        base = F.softplus(self.margin - signed_dist)  # (B,)

        if self.use_region_mask:
            mask = torch.exp(- (y_true - self.tau) ** 2 / (2 * self.sigma ** 2))  # (B,)
            if sample_mask is not None:
                mask = mask * sample_mask.to(dtype=mask.dtype, device=mask.device).reshape_as(mask)
            loss = (base * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = masked_mean(base, sample_mask)

        return loss





class EdgeBinnedCrossEntropyLoss(nn.Module):
    """
    Edge-level discretization + cross entropy loss.

    Supports:
    - optional error head
    - custom min/max range
    - optional edge mask
    """

    def __init__(
        self,
        num_error_bins: int,
        min_error: float,
        max_error: float,
        lambda_error: float,

        clamp: bool = True,
        ignore_index: int = -100,
    ):
        super().__init__()

        assert num_error_bins > 1
        assert max_error > min_error

        self.num_error_bins = num_error_bins
        self.min_error = min_error
        self.max_error = max_error
        self.lambda_error = lambda_error
        
        self.clamp = clamp
        self.ignore_index = ignore_index

    def forward(
        self,
        dist_error_pred: torch.Tensor,
        dist_error_label: torch.Tensor,
        edge_mask: torch.Tensor = None,
    ):
        """
        edge_mask: (N,) bool / {0,1}. Optional.
        """
        err_loss = self._cal_loss(
                label=dist_error_label,
                pred=dist_error_pred,
                num_bins=self.num_error_bins,
                min_val=self.min_error,
                max_val=self.max_error,
                edge_mask=edge_mask,
            )

        return self.lambda_error * err_loss


    def _cal_loss(
        self,
        label: torch.Tensor,   # (N,)
        pred: torch.Tensor,    # (N, num_bins)
        num_bins: int,
        min_val: float,
        max_val: float,
        edge_mask: torch.Tensor = None,
    ):
        """
        Continuous -> bin index -> cross entropy.
        """

        if self.clamp:
            label = torch.clamp(label, min_val, max_val)

        span = max_val - min_val
        if span <= 0:
            raise ValueError("max_val must be > min_val")

        # normalize to [0,1]
        x = (label - min_val) / span

        # bin index
        bin_index = torch.floor(x * num_bins).long()
        bin_index = torch.clamp(bin_index, 0, num_bins - 1)

        # apply mask
        if edge_mask is not None:
            if not edge_mask.bool().any():
                return pred.new_zeros(())
            bin_index = bin_index.masked_fill(
                ~edge_mask.bool(),
                self.ignore_index,
            )

        loss = F.cross_entropy(
            pred,
            bin_index,
            ignore_index=self.ignore_index,
            label_smoothing=0.05,
        )

        return loss




class EdgeBinnedSoftTargetLoss(nn.Module):
    """Cross-entropy on the error head with Gaussian soft targets over bins.

    Args:
        num_error_bins: number of bins for the error head output
        min_error, max_error: range covered by the bin grid (matches the head
            output dimensionality)
        sigma_bins: Gaussian std on the soft target, in units of bin widths
        clamp_label: if True, clamp out-of-range labels to [min, max] before
            building the soft target. If False, OOR samples get a degenerate
            but still valid distribution (most mass on the boundary bin).
        ignore_value: optional sentinel value in the label tensor that should
            be excluded from the loss (e.g. -999 for missing labels). Set to
            None to disable.

    Forward:
        error_pred: (E, num_error_bins) logits
        error_label: (E,) continuous error values

    Returns:
        loss: scalar tensor — mean cross-entropy with soft targets
              (= KL(soft_target || pred) up to an additive constant)
    """

    def __init__(self,
                 num_error_bins: int,
                 min_error: float,
                 max_error: float,
                 sigma_bins: float,
                 lambda_error: float,
                 clamp_label: bool = True,
                 ignore_value: float = None):
        super().__init__()
        assert num_error_bins >= 2
        assert max_error > min_error
        assert sigma_bins > 0.0

        self.num_error_bins = num_error_bins
        self.min_error = float(min_error)
        self.max_error = float(max_error)
        self.sigma_bins = float(sigma_bins)
        self.clamp_label = clamp_label
        self.ignore_value = ignore_value
        self.lambda_error = float(lambda_error)

        bin_width = (max_error - min_error) / num_error_bins
        # bin centers, length num_error_bins
        bin_edges = torch.linspace(min_error, max_error, num_error_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.register_buffer('bin_centers', bin_centers, persistent=False)
        # physical sigma (label units) = bin_width × sigma_bins
        self.sigma_physical = bin_width * sigma_bins

        print(f"[EdgeBinnedSoftTargetLoss] num_bins={num_error_bins} "
              f"range=[{min_error:.2f}, {max_error:.2f}] "
              f"bin_width={bin_width:.3f} "
              f"sigma_bins={sigma_bins} "
              f"sigma_physical={self.sigma_physical:.3f}")

    def make_soft_target(self, values: torch.Tensor) -> torch.Tensor:
        """Build (N, num_bins) Gaussian soft target distribution.

        Each row sums to 1 (softmax-normalised). For a label value v and bin
        center c_i, the unnormalised log-probability is -(v - c_i)^2 / (2σ²).
        Using softmax for normalisation avoids tail-truncation artefacts and
        keeps the result a proper distribution even when v is at the boundary.
        """
        if self.clamp_label:
            values = values.clamp(self.min_error, self.max_error)
        # (N, 1) - (1, B) -> (N, B)
        diff = values.unsqueeze(-1) - self.bin_centers.unsqueeze(0)
        log_p = -0.5 * (diff / self.sigma_physical) ** 2
        return F.softmax(log_p, dim=-1)

    def forward(self,
                error_pred: torch.Tensor,
                error_label: torch.Tensor) -> torch.Tensor:
        # Optional masking of sentinel-valued labels
        if self.ignore_value is not None:
            valid_mask = (error_label != self.ignore_value)
            if valid_mask.sum() == 0:
                # no valid samples this batch — return a zero loss that still
                # carries gradient flow shape (so optimizer.step() is a no-op)
                return error_pred.sum() * 0.0
            error_pred = error_pred[valid_mask]
            error_label = error_label[valid_mask]

        soft_target = self.make_soft_target(error_label)        # (E, B)
        log_pred = F.log_softmax(error_pred, dim=-1)            # (E, B)
        # cross-entropy with soft target = - Σ_b target_b * log_pred_b
        # mean over edges
        loss = -(soft_target * log_pred).sum(dim=-1).mean()
        return self.lambda_error * loss
