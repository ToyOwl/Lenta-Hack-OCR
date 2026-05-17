"""
TILT/skew correction for price-tag and shelf-rail OCR crops.
The implementation is based on the classic horizontal-projection criterion:
rotate the image by a set of candidate angles, binarize the text strokes, compute
row projections, and choose the angle with the highest projection variance.  Text
lines aligned with the X axis produce sharper projection peaks.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .pipeline_types import OCRItem


class TiltCorrector:
    def __init__(
        self,
        coarse_step: float = 1.0,
        fine_step: float = 0.2,
        angle_range: float = 35.0,
        min_text_height: int = 12,
        min_abs_angle: float = 1.2,
        max_work_side: int = 900,
        debug: bool = False,
    ) -> None:
        self.coarse_step = float(coarse_step)
        self.fine_step = float(fine_step)
        self.angle_range = float(angle_range)
        self.min_text_height = int(min_text_height)
        self.min_abs_angle = float(min_abs_angle)
        self.max_work_side = int(max_work_side)
        self.debug = bool(debug)
        self.last_debug: dict = {}

    def correct(self, image: np.ndarray) -> Tuple[np.ndarray, float]:

        """
        Return (deskewed_image, applied_angle_degrees)
        """

        self.last_debug = {"enabled": True, "applied": False, "angle": 0.0}

        if image is None or image.size == 0 or image.shape[0] < 32 or image.shape[1] < 64:
           self.last_debug["reason"] = "image_too_small"
           return image.copy(), 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        work_gray, scale = self._downscale_for_search(gray)

        best_angle = self._find_best_angle(work_gray)
        self.last_debug.update({"best_angle_raw": float(best_angle), "search_scale": float(scale)})

        if abs(best_angle) < self.min_abs_angle:
            self.last_debug["reason"] = "angle_below_threshold"
            return image.copy(), 0.0

        rotated = self._rotate(image, best_angle)
        self.last_debug.update({"applied": True, "angle": float(best_angle)})
        return rotated, float(best_angle)

    def _downscale_for_search(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:

        h, w = gray.shape[:2]
        side = max(h, w)

        if self.max_work_side <= 0 or side <= self.max_work_side:
            return gray, 1.0

        scale = self.max_work_side / float(side)
        out = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

        return out, float(scale)

    def _find_best_angle(self, gray: np.ndarray) -> float:

        angles_coarse = np.arange(-self.angle_range, self.angle_range + 1e-6, self.coarse_step, dtype=np.float32)
        scores_coarse = np.asarray([self._projection_variance(gray, float(ang)) for ang in angles_coarse], dtype=np.float32)
        best_idx = int(np.argmax(scores_coarse)) if scores_coarse.size else 0
        best_angle_coarse = float(angles_coarse[best_idx]) if angles_coarse.size else 0.0

        fine_lo = best_angle_coarse - self.coarse_step
        fine_hi = best_angle_coarse + self.coarse_step
        angles_fine = np.arange(fine_lo, fine_hi + 1e-6, self.fine_step, dtype=np.float32)
        scores_fine = np.asarray([self._projection_variance(gray, float(ang)) for ang in angles_fine], dtype=np.float32)
        best_idx_fine = int(np.argmax(scores_fine)) if scores_fine.size else 0
        best_angle = float(angles_fine[best_idx_fine]) if angles_fine.size else best_angle_coarse

        if self.debug:
          self.last_debug["coarse"] = {"best_angle": best_angle_coarse, "best_score": float(scores_coarse[best_idx]) if scores_coarse.size else 0.0,}
          self.last_debug["fine"] = {"best_angle": best_angle, "best_score": float(scores_fine[best_idx_fine]) if scores_fine.size else 0.0,}

        return best_angle

    def _projection_variance(self, gray: np.ndarray, angle: float) -> float:
       rotated = self._rotate(gray, angle)

       if rotated.size == 0:
         return 0.0

       blur = cv2.GaussianBlur(rotated, (3, 3), 0)
       _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

       # Suppress black triangular borders introduced by rotation. Otherwise
       # they can create artificial projection peaks for large angles.
       h, w = binary.shape[:2]
       pad_y = max(1, int(h * 0.015))
       pad_x = max(1, int(w * 0.015))
       if h > 2 * pad_y and w > 2 * pad_x:
          binary[:pad_y, :] = 0
          binary[-pad_y:, :] = 0
          binary[:, :pad_x] = 0
          binary[:, -pad_x:] = 0

       # Remove tiny isolated noise but keep character strokes.
       k = max(1, int(round(self.min_text_height / 10)))
       kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, k), max(1, k)))
       binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

       proj = np.sum(binary > 0, axis=1).astype(np.float32)

       if float(np.ptp(proj)) < 1e-6 or float(np.sum(proj)) < 10.0:
          return 0.0

       # Normalize by foreground amount to reduce preference for angles that
       # accidentally create more border/foreground pixels.
       return float(np.var(proj) / (np.mean(proj) + 1e-3))

    @staticmethod
    def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
       h, w = image.shape[:2]
       center = (w / 2.0, h / 2.0)
       rot_mat = cv2.getRotationMatrix2D(center, float(angle), 1.0)

       if image.ndim == 2:
          border_value = int(np.median(image))

       else:
            med = np.median(image.reshape(-1, image.shape[-1]), axis=0)
            border_value = tuple(int(x) for x in med.tolist())

       return cv2.warpAffine(image, rot_mat,(w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,)

    def correct_ocr_items(self, items: List[OCRItem], angle: float, orig_h: int, orig_w: int) -> List[OCRItem]:
        """Rotate OCR polygons by the same transform used for image deskewing."""

        if abs(angle) < 1.0:
            return items

        rot_mat = cv2.getRotationMatrix2D((orig_w / 2.0, orig_h / 2.0), float(angle), 1.0)
        corrected: List[OCRItem] = []

        for item in items:

          if item.box is None:
             corrected.append(item)
             continue

          pts = np.asarray(item.box, dtype=np.float32).reshape(-1, 1, 2)
          rotated_pts = cv2.transform(pts, rot_mat)
          new_box = rotated_pts.reshape(-1, 2).tolist()
          corrected.append(OCRItem(text=item.text, conf=item.conf, box=new_box, zone=item.zone))
        return corrected

    def get_debug_info(self) -> dict:
        return dict(self.last_debug)


if __name__ == "__main__":

  import sys
  from pathlib import Path

  from .io_utils import imread_unicode, imwrite_unicode

  if len(sys.argv) > 1:
     path = Path(sys.argv[1])
     img = imread_unicode(path)

     if img is not None:
        corrector = TiltCorrector(debug=True)
        fixed, angle = corrector.correct(img)
        print(f"TILT: angle = {angle:.2f}°")
        imwrite_unicode(path.with_name(path.stem + "_tilt_fixed.jpg"), fixed)
