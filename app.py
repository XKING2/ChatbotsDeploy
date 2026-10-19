import os
import uuid

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    g,
)
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

import config
from LLM.chatbot import ImmigrationChatbot
from models import db, Admin, Conversation, Message
from knowledge_base import ingest_document
from datetime import timezone
from zoneinfo import ZoneInfo

WITA = ZoneInfo("Asia/Makassar")


app = Flask(__name__)
_DEFAULT_SECRET = "ganti-dengan-secret-key-acak"
app.secret_key = os.environ.get("FLASK_SECRET_KEY", _DEFAULT_SECRET)
if app.secret_key == _DEFAULT_SECRET:
    app.logger.warning(
        "FLASK_SECRET_KEY belum di-set -- pakai secret key bawaan yang TIDAK aman. "
        "Wajib di-set di server publik (Railway)."
    )

# ============================================================
# DATABASE
# ============================================================

# URL database & folder upload diatur di config.py:
#   - Railway : DATABASE_URL=sqlite:////data/chat.db , DATA_DIR=/data
#   - Laptop  : tanpa env -> MySQL dari DB_* seperti sebelumnya
app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

UPLOAD_FOLDER = config.UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Batas ukuran upload PDF (RAM server gratis terbatas).
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

db.init_app(app)

# Dengan gunicorn blok `if __name__ == "__main__"` di bawah TIDAK jalan,
# jadi tabel harus dibuat di sini (aman dijalankan berulang).
with app.app_context():
    db.create_all()


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "message": "Ukuran file terlalu besar (maksimal 25 MB).",
    }), 413

# ============================================================
# FLASK-LOGIN: ADMIN ONLY
# ============================================================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "admin_login"
login_manager.login_message = "Silakan login sebagai admin terlebih dahulu."


@login_manager.user_loader
def load_user(admin_id):
    # Query.get masih kompatibel dengan versi Flask-SQLAlchemy lama.
    return db.session.get(Admin, int(admin_id))


# RetrievalPipeline is owned by ImmigrationChatbot; do not run a second
# semantic_search from this Flask layer.
chatbot = ImmigrationChatbot()


# ============================================================
# PUBLIC SESSION
# ============================================================

@app.before_request
def ensure_session_cookie():
    if not request.cookies.get("session_id"):
        g.new_session_id = str(uuid.uuid4())


@app.after_request
def set_session_cookie(response):
    if hasattr(g, "new_session_id"):
        response.set_cookie(
            "session_id",
            g.new_session_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="Lax",
        )
    return response


def current_session_id():
    if hasattr(g, "new_session_id"):
        return g.new_session_id
    return request.cookies.get("session_id")


# ============================================================
# PUBLIC CHAT
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    session_id = current_session_id()
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id")

    if not question:
        return jsonify({"error": "Pesan tidak boleh kosong"}), 400

    # Conversation IDs are scoped to the current browser session.
    conversation = None
    if conversation_id is not None:
        try:
            conversation_id = int(conversation_id)
        except (TypeError, ValueError):
            return jsonify({"error": "ID percakapan tidak valid"}), 400

        conversation = Conversation.query.filter_by(
            id=conversation_id,
            session_id=session_id,
        ).first()

    if conversation is None:
        title = question[:50] + ("..." if len(question) > 50 else "")
        conversation = Conversation(
            session_id=session_id,
            title=title,
        )
        db.session.add(conversation)
        db.session.commit()

    previous_entity = conversation.last_entity
    previous_question = conversation.last_question

    db.session.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=question,
        )
    )
    db.session.commit()

    try:
        # LLM.chatbot handles the complete flow:
        # question -> RetrievalPipeline -> relevance gate -> JSON context
        # -> prompt -> LLM. Flask must not perform retrieval itself.
        result = chatbot.ask(
            question=question,
            previous_entity=previous_entity,
            previous_question=previous_question,
        )
    except Exception:
        app.logger.exception("Gagal memproses pertanyaan chatbot")
        return jsonify({
            "error": "Maaf, terjadi kesalahan saat memproses pertanyaan. Silakan coba lagi."
        }), 500

    answer = result.get(
        "answer",
        "Maaf, jawaban belum dapat dibuat saat ini.",
    )
    found = bool(result.get("found", False))
    entity = result.get("entity")

    # Prevent an old entity from being reused after an unrelated/unsupported
    # question. For a valid follow-up, the pipeline should resolve and return
    # the current entity itself.
    conversation.last_entity = entity if found else None
    conversation.last_question = result.get("effective_question", question)

    bot_message = Message(
        conversation_id=conversation.id,
        role="bot",
        content=answer,
    )
    db.session.add(bot_message)
    db.session.commit()

    return jsonify({
    "conversation_id": conversation.id,
    "answer": answer,
    "time": format_wita(bot_message.created_at),
    "found": found,
    "status": result.get("status"),
})


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    session_id = current_session_id()
    conversations = (
        Conversation.query.filter_by(session_id=session_id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    return jsonify([conversation.to_dict() for conversation in conversations])


@app.route("/api/conversations/<int:conversation_id>/messages", methods=["GET"])
def get_conversation_messages(conversation_id):
    session_id = current_session_id()
    conversation = Conversation.query.filter_by(
        id=conversation_id,
        session_id=session_id,
    ).first()

    if conversation is None:
        return jsonify({"error": "Percakapan tidak ditemukan"}), 404

    return jsonify([message.to_dict() for message in conversation.messages])

def format_wita(dt):
    if dt is None:
        return None

    # Timestamp database yang naive diasumsikan tersimpan dalam UTC.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(WITA).strftime("%H.%M")


# ============================================================
# ADMIN: KNOWLEDGE BASE MANAGEMENT
# ============================================================

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if current_user.is_authenticated:
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        admin = Admin.query.filter_by(email=email).first()

        if admin is None or not check_password_hash(admin.password_hash, password):
            flash("Email atau password salah.")
            return redirect(url_for("admin_login"))

        login_user(admin)
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_login.html")


@app.route("/admin/logout")
@login_required
def admin_logout():
    logout_user()
    return redirect(url_for("admin_login"))


@app.route("/adminpage")
@login_required
def admin_dashboard():
    return render_template("adminpage.html")


@app.route("/admin/upload-dokumen", methods=["POST"])
@login_required
def admin_upload_dokumen():
    file = request.files.get("document")

    if file is None or file.filename == "":
        return jsonify({
            "success": False,
            "message": "Pilih file PDF terlebih dahulu.",
        }), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({
            "success": False,
            "message": "File harus berformat PDF.",
        }), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({
            "success": False,
            "message": "Nama file tidak valid.",
        }), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(file_path)
        result = ingest_document(file_path)

        if result.get("success"):
            return jsonify({
                "success": True,
                "message": result.get(
                    "message",
                    "Knowledge base berhasil diperbarui.",
                ),
            })

        return jsonify({
            "success": False,
            "message": result.get(
                "message",
                "Gagal melakukan ingest dokumen.",
            ),
        }), 500

    except Exception:
        app.logger.exception("Gagal mengunggah/meng-ingest dokumen")
        return jsonify({
            "success": False,
            "message": "Terjadi kesalahan saat memproses dokumen.",
        }), 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)