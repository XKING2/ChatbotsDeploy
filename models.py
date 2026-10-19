from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class Admin(UserMixin, db.Model):
    __tablename__ = "admins"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Conversation(db.Model):
    __tablename__ = "conversations"

    id = db.Column(db.Integer, primary_key=True)
    # session_id = identitas anonim per browser/device (dari Flask session,
    # dibuat otomatis saat pengunjung pertama kali buka halaman)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    title = db.Column(db.String(255), nullable=False, default="Percakapan baru")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # State context untuk ImmigrationChatbot.ask(), disimpan PER
    # CONVERSATION supaya tiap thread riwayat punya context sendiri.
    last_entity = db.Column(db.String(255), nullable=True)
    last_question = db.Column(db.Text, nullable=True)

    messages = db.relationship(
        "Message",
        backref="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.strftime("%H.%M"),
        }


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(
        db.Integer, db.ForeignKey("conversations.id"), nullable=False, index=True
    )
    role = db.Column(db.String(10), nullable=False)  # "user" atau "bot"
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "time": self.created_at.strftime("%H.%M"),
        }