
import os
import sys
import getpass

from werkzeug.security import generate_password_hash
from app import app
from models import db, Admin


def main():
    # Ambil kredensial dari environment Railway jika tersedia.
    email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    password = os.getenv("ADMIN_PASSWORD", "")

    # Mode interaktif untuk penggunaan lokal.
    if not email:
        if not sys.stdin.isatty():
            print("ERROR: ADMIN_EMAIL belum diatur.")
            sys.exit(1)

        email = input("Email admin: ").strip().lower()

    if not password:
        if not sys.stdin.isatty():
            print("ERROR: ADMIN_PASSWORD belum diatur.")
            sys.exit(1)

        password = getpass.getpass("Password admin: ")

        confirm = getpass.getpass("Konfirmasi password: ")
        if password != confirm:
            print("Password tidak sama, dibatalkan.")
            sys.exit(1)

    if not email:
        print("Email admin wajib diisi.")
        sys.exit(1)

    if len(password) < 6:
        print("Password minimal 6 karakter.")
        sys.exit(1)

    with app.app_context():
        db.create_all()

        existing_admin = Admin.query.filter_by(email=email).first()

        if existing_admin:
            print(f"Admin dengan email '{email}' sudah ada.")
            return

        admin = Admin(
            email=email,
            password_hash=generate_password_hash(password),
        )

        try:
            db.session.add(admin)
            db.session.commit()
            print(f"Admin '{email}' berhasil dibuat.")

        except Exception:
            db.session.rollback()
            raise


if __name__ == "__main__":
    main()
