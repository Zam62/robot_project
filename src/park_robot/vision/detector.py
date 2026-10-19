"""Поиск черной линии на светлой поверхности. Камера и моторы здесь не используются."""

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np


@dataclass(frozen=True)
class VisionSettings:
    """Настройки обработки; доли площади не зависят от разрешения камеры."""

    roi_top: float = 0.55
    roi_bottom: float = 0.92
    blur_kernel: int = 5
    threshold_mode: str = "otsu"
    threshold_value: int = 100
    min_contour_area_ratio: float = 0.01
    max_contour_area_ratio: float = 0.65
    dead_zone_normalized: float = 0.03
    min_contrast: int = 30
    min_line_height_ratio: float = 0.5

    def __post_init__(self):
        fractions = (
            self.roi_top,
            self.roi_bottom,
            self.min_contour_area_ratio,
            self.max_contour_area_ratio,
            self.dead_zone_normalized,
            self.min_line_height_ratio,
        )
        if not all(isinstance(value, (int, float)) and isfinite(value) for value in fractions):
            raise ValueError("Доли и отклонения должны быть конечными числами")
        if not 0 <= self.roi_top < self.roi_bottom <= 1:
            raise ValueError("Нужно 0 <= roi_top < roi_bottom <= 1")
        if type(self.blur_kernel) is not int or self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel должен быть положительным нечетным целым числом")
        if self.threshold_mode not in ("otsu", "fixed"):
            raise ValueError("threshold_mode: используйте otsu или fixed")
        for value in (self.threshold_value, self.min_contrast):
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError("Порог и минимальный контраст: целые числа от 0 до 255")
        if not 0 < self.min_contour_area_ratio < self.max_contour_area_ratio <= 1:
            raise ValueError("Нужно 0 < min_contour_area_ratio < max_contour_area_ratio <= 1")
        if not 0 <= self.dead_zone_normalized < 1:
            raise ValueError("dead_zone_normalized должна быть от 0 включительно до 1")
        if not 0 < self.min_line_height_ratio <= 1:
            raise ValueError("min_line_height_ratio должна быть больше 0 и не больше 1")


@dataclass
class LineResult:
    """Результат одного кадра. При found=False ошибки равны None, а не нулю."""

    found: bool
    center: tuple[float, float] | None
    error_pixels: float | None
    error_normalized: float | None
    reason: str
    mask: np.ndarray
    contour: np.ndarray | None
    roi_top: int
    roi_bottom: int


def detect_line(frame: np.ndarray, settings: VisionSettings) -> LineResult:
    """Вернуть центр линии и отклонение: отрицательное влево, положительное вправо."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] < 2
        or frame.shape[1] < 2
    ):
        raise ValueError("Ожидается непустой BGR-кадр uint8 с тремя каналами")

    height, width = frame.shape[:2]
    top = int(height * settings.roi_top)
    bottom = int(height * settings.roi_bottom)
    if bottom <= top:
        raise ValueError("Область поиска пуста: измените ROI или разрешение")
    roi = frame[top:bottom, :]

    # В маске белыми будут пиксели темной линии.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (settings.blur_kernel, settings.blur_kernel), 0)
    threshold_type = cv2.THRESH_BINARY_INV
    threshold_value = settings.threshold_value
    if settings.threshold_mode == "otsu":
        threshold_type |= cv2.THRESH_OTSU
        threshold_value = 0
    _, mask = cv2.threshold(blurred, threshold_value, 255, threshold_type)

    result = LineResult(
        found=False,
        center=None,
        error_pixels=None,
        error_normalized=None,
        reason="no_line",
        mask=mask,
        contour=None,
        roi_top=top,
        roi_bottom=bottom,
    )

    # На почти однотонном кадре нельзя уверенно отличить линию от фона.
    contrast = int(blurred.max()) - int(blurred.min())
    if contrast < settings.min_contrast:
        result.reason = "low_contrast"
        return result
    dark_ratio = cv2.countNonZero(mask) / mask.size
    if dark_ratio > settings.max_contour_area_ratio:
        result.reason = "too_dark"
        return result

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area_ratio = cv2.contourArea(contour) / mask.size
        _, _, _, contour_height = cv2.boundingRect(contour)
        enough_height = contour_height >= mask.shape[0] * settings.min_line_height_ratio
        if (
            settings.min_contour_area_ratio <= area_ratio <= settings.max_contour_area_ratio
            and enough_height
        ):
            candidates.append(contour)

    if not candidates:
        return result
    if len(candidates) > 1:
        result.reason = "ambiguous"
        return result

    contour = candidates[0]
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        result.reason = "zero_area"
        return result

    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"] + top
    error_pixels = center_x - width / 2
    error_normalized = error_pixels / (width / 2)
    if abs(error_normalized) < settings.dead_zone_normalized:
        error_normalized = 0.0

    result.found = True
    result.center = (center_x, center_y)
    result.error_pixels = error_pixels
    result.error_normalized = error_normalized
    result.reason = "found"
    result.contour = contour
    return result
