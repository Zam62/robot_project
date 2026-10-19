"""Демонстрация зрения без двигателей; можно работать без графического дисплея."""

import csv
import tomllib
from pathlib import Path

import cv2

from park_robot.camera.opencv_source import open_source
from park_robot.vision.detector import VisionSettings, detect_line
from park_robot.vision.overlay import draw_result


def run_demo(
    source: int | str,
    config_path: Path,
    headless: bool = False,
    output_dir: Path | None = None,
    max_frames: int | None = None,
) -> int:
    """Обработать источник, показать результат и при необходимости сохранить отчет."""
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames должен быть положительным")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    settings = VisionSettings(**config.get("vision", {}))
    camera = config.get("camera", {})
    capture = None
    report_file = None
    frame_count = 0
    found_count = 0
    previous_found = None

    try:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            report_file = (output_dir / "results.csv").open("w", newline="", encoding="utf-8")
            report = csv.writer(report_file)
            report.writerow(
                ["frame", "found", "center_x", "error_pixels", "error_normalized", "reason"]
            )
        capture = open_source(
            source, camera.get("width", 640), camera.get("height", 480), camera.get("fps", 30)
        )
        while max_frames is None or frame_count < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                if isinstance(source, int) or frame_count == 0:
                    raise RuntimeError("Не удалось получить кадр; демонстрация остановлена")
                # Для видеофайла read=False после кадров обычно означает конец записи.
                break

            result = detect_line(frame, settings)
            preview = draw_result(frame, result)
            status_changed = result.found != previous_found
            if status_changed:
                if result.found:
                    print("Линия найдена.", flush=True)
                else:
                    print(f"Линия не найдена. Причина: {result.reason}.", flush=True)
                previous_found = result.found
            frame_count += 1
            found_count += int(result.found)
            if report_file is not None:
                center_x = result.center[0] if result.found else ""
                report.writerow(
                    [
                        frame_count,
                        result.found,
                        center_x,
                        result.error_pixels,
                        result.error_normalized,
                        result.reason,
                    ]
                )
                # Два последних изображения, а не вся запись, чтобы не заполнять диск.
                for name, image in (("preview.png", preview), ("mask.png", result.mask)):
                    if not cv2.imwrite(str(output_dir / name), image):
                        raise RuntimeError(f"Не удалось сохранить {name}")

            if not headless:
                cv2.imshow("Park Robot - camera", preview)
                if status_changed:
                    status = "ЛИНИЯ НАЙДЕНА" if result.found else "ЛИНИЯ НЕ НАЙДЕНА"
                    cv2.setWindowTitle("Park Robot - camera", f"Робот: {status}")
                cv2.imshow("Park Robot - line mask", result.mask)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("Park Robot - camera", cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        try:
            if capture is not None:
                capture.release()
        finally:
            try:
                if report_file is not None:
                    report_file.close()
            finally:
                if not headless:
                    cv2.destroyAllWindows()

    print(f"Обработано кадров: {frame_count}. Линия найдена: {found_count}.")
    print("Режим демонстрации: двигатели не подключались.")
    return 0
