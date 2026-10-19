"""Отладочное изображение: область поиска, контур, центр и отклонение."""

import cv2
import numpy as np

from park_robot.vision.detector import LineResult


def draw_result(frame: np.ndarray, result: LineResult) -> np.ndarray:
    """Нарисовать результат на копии, не меняя исходный кадр."""
    preview = frame.copy()
    width = frame.shape[1]
    cv2.rectangle(preview, (0, result.roi_top), (width - 1, result.roi_bottom - 1), (255, 0, 0), 2)
    cv2.line(
        preview,
        (width // 2, result.roi_top),
        (width // 2, result.roi_bottom - 1),
        (0, 0, 255),
        2,
    )

    if result.found:
        roi = preview[result.roi_top : result.roi_bottom, :]
        cv2.drawContours(roi, [result.contour], -1, (0, 255, 0), 2)
        center = (round(result.center[0]), round(result.center[1]))
        cv2.circle(preview, center, 6, (0, 255, 255), -1)
        direction = "CENTER"
        if result.error_normalized < 0:
            direction = "LEFT"
        elif result.error_normalized > 0:
            direction = "RIGHT"
        lines = [
            direction,
            f"error={result.error_pixels:+.1f}px",
            f"norm={result.error_normalized:+.3f}",
        ]
        color = (0, 255, 0)
    else:
        lines = [f"STOP: {result.reason}"]
        color = (0, 0, 255)

    # Встроенный шрифт OpenCV поддерживает латиницу; русский статус есть в заголовке окна.
    status = "LINE FOUND" if result.found else "LINE NOT FOUND"
    background = (0, 110, 0) if result.found else (0, 0, 160)
    cv2.rectangle(preview, (0, 0), (width - 1, 39), background, -1)
    cv2.putText(preview, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for number, text in enumerate(lines):
        cv2.putText(preview, text, (10, 62 + number * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return preview
