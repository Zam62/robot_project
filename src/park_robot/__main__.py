"""Диагностика окружения и демонстрация компьютерного зрения без двигателей."""

import argparse
import importlib
import json
import platform
import sys
import tomllib
from pathlib import Path


def doctor(config_path: Path) -> dict:
    checks = {}
    for name in ("cv2", "numpy", "tkinter"):
        try:
            module = importlib.import_module(name)
            checks[name] = {
                "ok": True,
                "version": str(getattr(module, "__version__", getattr(module, "TkVersion", ""))),
            }
        except (ImportError, OSError) as exc:
            checks[name] = {"ok": False, "error": str(exc)}
    try:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
        checks["config"] = {
            "ok": True,
            "path": str(config_path),
            "motor_backend": config.get("motors", {}).get("backend", "unspecified"),
            "note": "TOML parsed; runtime settings are not validated yet",
        }
    except (OSError, tomllib.TOMLDecodeError) as exc:
        checks["config"] = {"ok": False, "error": str(exc)}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checks": checks,
        "ok": all(item["ok"] for item in checks.values()),
        "scope": "Imports and TOML syntax only; no camera, GUI display, GPIO or motor checks",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnostics = subparsers.add_parser("doctor", help="Проверить импорты и синтаксис TOML")
    diagnostics.add_argument("--config", type=Path, default=Path("configs/desktop.toml"))
    diagnostics.add_argument("--json", action="store_true")
    demo = subparsers.add_parser("demo", help="Найти линию на камере или видео без моторов")
    sources = demo.add_mutually_exclusive_group(required=True)
    sources.add_argument("--camera", type=int, help="Номер USB-камеры, например 0")
    sources.add_argument("--video", type=Path, help="Путь к видеофайлу")
    demo.add_argument("--config", type=Path, default=Path("configs/desktop.toml"))
    demo.add_argument("--headless", action="store_true", help="Не открывать окна")
    demo.add_argument("--output", type=Path, help="Папка для CSV, preview.png и mask.png")
    demo.add_argument("--max-frames", type=int, help="Ограничить количество кадров")
    args = parser.parse_args()
    if args.command == "demo":
        try:
            # Импорт нужен только для demo: doctor работает даже без установленного OpenCV.
            from park_robot.vision.demo import run_demo

            source = args.camera if args.camera is not None else str(args.video)
            return run_demo(source, args.config, args.headless, args.output, args.max_frames)
        except KeyboardInterrupt:
            print("Демонстрация прервана пользователем.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"Ошибка демонстрации: {exc}", file=sys.stderr)
            return 1
    report = doctor(args.config)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Python {report['python']}; {report['platform']}")
        for name, result in report["checks"].items():
            print(f"{name}: {'OK' if result['ok'] else result['error']}")
        print(report["scope"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
