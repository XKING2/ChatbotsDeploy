

import getpass
from werkzeug.security import generate_password_hash

from app import app
from models import db, Admin


def main():
    email = input("Email admin: ").strip().lower()
    password = getpass.getpass("Password admin: ")
    confirm = getpass.getpass("Konfirmasi password: ")

    if password != confirm:
        print("Password tidak sama, dibatalkan.")
        return

    if len(password) < 6:
        print("Password minimal 6 karakter, dibatalkan.")
        return

    with app.app_context():
        db.create_all()

        if Admin.query.filter_by(email=email).first():
            print(f"Admin dengan email '{email}' sudah ada.")
            return

        admin = Admin(email=email, password_hash=generate_password_hash(password))
        db.session.add(admin)
        db.session.commit()

        print(f"Admin '{email}' berhasil dibuat.")


if __name__ == "__main__":
    main()