import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MAIN_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "database",
    "main.py",
)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("MAIN_SCRIPT:", MAIN_SCRIPT)
print("MAIN_SCRIPT EXISTS:", os.path.isfile(MAIN_SCRIPT))

MAIN_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "database",
    "main.py",
)


def ingest_document(file_path: str) -> dict:
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"

    # Agar main.py bisa menemukan config.py di root proyek.
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (PROJECT_ROOT, existing_pythonpath)
        if path
    )

    try:
        process = subprocess.run(
            [
                sys.executable,
                "-u",
                MAIN_SCRIPT,
                os.path.abspath(file_path),
            ],
            cwd=PROJECT_ROOT,
            env=child_env,
            text=True,
            encoding="utf-8",
            check=False,
        )

        if process.returncode != 0:
            return {
                "success": False,
                "message": (
                    "Gagal update knowledge base. "
                    "Periksa traceback pada log server."
                ),
            }

        return {
            "success": True,
            "message": "Knowledge base berhasil diupdate.",
        }

    except OSError:
        import logging
        logging.exception("Gagal menjalankan proses ingestion")

        return {
            "success": False,
            "message": "Gagal menjalankan proses ingestion.",
        }