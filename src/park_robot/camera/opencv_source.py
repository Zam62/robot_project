"""Открытие USB-камеры или видеофайла через OpenCV."""

import cv2


def open_source(source: int | str, width: int, height: int, fps: float) -> cv2.VideoCapture:
    """Вернуть открытый источник. Освобождение через release выполняет вызывающий код."""
    if isinstance(source, int) and source < 0:
        raise ValueError("Номер камеры должен быть неотрицательным")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("Ширина, высота и FPS должны быть положительными")

    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {source}")
        if isinstance(source, int):
            # Камера может выбрать другие параметры; детектор использует реальный размер кадра.
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
        return capture
    except Exception:
        capture.release()
        raise
