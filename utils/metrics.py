"""
Evaluation Metrics for Camouflaged Object Detection.

Implements standard COD metrics:
  - MAE (Mean Absolute Error)
  - S-measure (Structure Measure)
  - E-measure (Enhanced-alignment Measure)
  - Weighted F-measure
"""
import numpy as np


class MetricCalculator:
    """Accumulates predictions and computes COD metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mae_list = []
        self.sm_list = []
        self.em_list = []
        self.fm_list = []

    def update(self, pred, gt):
        """
        Update metrics with a single prediction-GT pair.

        Args:
            pred: Predicted mask [H, W], float in [0, 1]
            gt:   Ground truth mask [H, W], binary {0, 1}
        """
        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"

        # Ensure numpy
        if hasattr(pred, "cpu"):
            pred = pred.cpu().numpy()
        if hasattr(gt, "cpu"):
            gt = gt.cpu().numpy()

        pred = pred.astype(np.float64)
        gt = gt.astype(np.float64)

        # MAE
        self.mae_list.append(np.mean(np.abs(pred - gt)))

        # S-measure
        self.sm_list.append(self._s_measure(pred, gt))

        # E-measure
        self.em_list.append(self._e_measure(pred, gt))

        # Weighted F-measure
        self.fm_list.append(self._weighted_f_measure(pred, gt))

    def get_results(self):
        """Return averaged metrics."""
        return {
            "MAE": np.mean(self.mae_list) if self.mae_list else 0.0,
            "S-measure": np.mean(self.sm_list) if self.sm_list else 0.0,
            "E-measure": np.mean(self.em_list) if self.em_list else 0.0,
            "wF-measure": np.mean(self.fm_list) if self.fm_list else 0.0,
        }

    def _s_measure(self, pred, gt, alpha=0.5):
        """Structure Measure (Fan et al., 2017)."""
        gt_mean = gt.mean()
        if gt_mean == 0:  # No foreground
            score = 1.0 - pred.mean()
        elif gt_mean == 1:  # All foreground
            score = pred.mean()
        else:
            s_obj = self._s_object(pred, gt)
            s_reg = self._s_region(pred, gt)
            score = alpha * s_obj + (1 - alpha) * s_reg
        return max(0.0, score)

    def _s_object(self, pred, gt):
        """Object-level structure similarity."""
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        fg_fit = self._object_score(fg, gt)
        bg_fit = self._object_score(bg, 1 - gt)

        u = gt.mean()
        return u * fg_fit + (1 - u) * bg_fit

    def _object_score(self, pred_region, gt_region):
        """Object score for a region."""
        x = pred_region[gt_region > 0.5]
        if len(x) == 0:
            return 0.0
        mu = x.mean()
        sigma = x.std()
        if mu == 0:
            return 0.0
        score = 2.0 * mu / (mu * mu + 1.0 + sigma + 1e-8)
        return score

    def _s_region(self, pred, gt):
        """Region-level structure similarity."""
        x, y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divide_gt(gt, x, y)
        p1, p2, p3, p4, _, _, _, _ = self._divide_gt(pred, x, y)

        q1 = self._ssim(p1, gt1)
        q2 = self._ssim(p2, gt2)
        q3 = self._ssim(p3, gt3)
        q4 = self._ssim(p4, gt4)

        return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4

    def _centroid(self, gt):
        h, w = gt.shape
        if gt.sum() == 0:
            return h // 2, w // 2
        rows, cols = np.where(gt > 0.5)
        return int(rows.mean()), int(cols.mean())

    def _divide_gt(self, matrix, x, y):
        h, w = matrix.shape
        lt = matrix[:x, :y]
        rt = matrix[:x, y:]
        lb = matrix[x:, :y]
        rb = matrix[x:, y:]

        area = h * w + 1e-8
        w1 = (lt.size) / area
        w2 = (rt.size) / area
        w3 = (lb.size) / area
        w4 = (rb.size) / area

        return lt, rt, lb, rb, w1, w2, w3, w4

    def _ssim(self, pred, gt):
        h, w = pred.shape
        n = h * w
        if n < 2:
            return 0.0

        x = pred.mean()
        y = gt.mean()
        sigma_x = pred.std()
        sigma_y = gt.std()
        sigma_xy = ((pred - x) * (gt - y)).mean()

        alpha = (4 * x * y * sigma_xy) / (
            (x * x + y * y) * (sigma_x * sigma_x + sigma_y * sigma_y) + 1e-8
        )
        return max(0.0, alpha)

    def _e_measure(self, pred, gt):
        """Enhanced-alignment Measure (Fan et al., 2018)."""
        th = pred.mean() * 2  # Adaptive threshold
        binary_pred = (pred >= th).astype(np.float64)

        gt_mean = gt.mean()
        if gt_mean == 0:
            enhanced = 1.0 - binary_pred.mean()
        elif gt_mean == 1:
            enhanced = binary_pred.mean()
        else:
            align = self._alignment_term(binary_pred, gt)
            enhanced = align.mean()

        return enhanced

    def _alignment_term(self, pred, gt):
        mu_pred = pred.mean()
        mu_gt = gt.mean()

        align_pred = pred - mu_pred
        align_gt = gt - mu_gt

        align_mat = (
            2.0 * align_gt * align_pred
            / (align_gt * align_gt + align_pred * align_pred + 1e-8)
        )
        enhanced = ((align_mat + 1.0) ** 2) / 4.0
        return enhanced

    def _weighted_f_measure(self, pred, gt, beta2=1.0):
        """Weighted F-measure."""
        eps = 1e-8
        # Adaptive threshold
        th = pred.mean() * 2
        binary_pred = (pred >= th).astype(np.float64)

        tp = (binary_pred * gt).sum()
        fp = (binary_pred * (1 - gt)).sum()
        fn = ((1 - binary_pred) * gt).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f_measure = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
        return f_measure
