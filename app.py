
import os
import uuid
from datetime import timezone
from zoneinfo import ZoneInfo

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


# ============================================================
# CONFIGURATION
# ============================================================

WITA = ZoneInfo("Asia/Makassar")

app = Flask(__name__)

_DEFAULT_SECRET = "ganti-dengan-secret-key-acak"
app.secret_key = os.environ.get("FLASK_SECRET_KEY", _DEFAULT_SECRET)

if app.secret_key == _DEFAULT_SECRET:
    app.logger.warning(
        "FLASK_SECRET_KEY belum diatur. "
        "Gunakan secret key acak melalui environment variable."
    )

app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

UPLOAD_FOLDER = config.UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# DATABASE
# ============================================================

db.init_app(app)

with app.app_context():
    db.create_all()


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(_error):
    if request.path == "/admin/upload-dokumen":
        flash(
            "Ukuran file terlalu besar. Maksimal 25 MB.",
            "error",
        )
        return redirect(url_for("admin_dashboard"))

    return jsonify({
        "success": False,
        "message": "Ukuran file terlalu besar (maksimal 25 MB).",
    }), 413


# ============================================================
# FLASK-LOGIN
# ============================================================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "admin_login"
login_manager.login_message = "Silakan login sebagai admin terlebih dahulu."
login_manager.login_message_category = "warning"


@login_manager.user_loader
def load_user(admin_id):
    try:
        return db.session.get(Admin, int(admin_id))
    except (TypeError, ValueError):
        return None


# ============================================================
# CHATBOT INITIALIZATION
# ============================================================

# Retrieval dan generation tetap ditangani oleh ImmigrationChatbot.
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
# TIME FORMAT
# ============================================================

def format_wita(dt):
    if dt is None:
        return None

    # Timestamp naive diasumsikan disimpan sebagai UTC.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(WITA).strftime("%H.%M")


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

    question = data.get("message")

    if not isinstance(question, str) or not question.strip():
        return jsonify({
            "error": "Pesan tidak boleh kosong."
        }), 400

    question = question.strip()
    conversation_id = data.get("conversation_id")

    try:
        # Cari percakapan yang dimiliki sesi browser ini.
        conversation = None

        if conversation_id is not None:
            try:
                conversation_id = int(conversation_id)
            except (TypeError, ValueError):
                return jsonify({
                    "error": "ID percakapan tidak valid."
                }), 400

            conversation = Conversation.query.filter_by(
                id=conversation_id,
                session_id=session_id,
            ).first()

        # Buat percakapan baru jika ID tidak diberikan
        # atau percakapan tidak dimiliki sesi ini.
        if conversation is None:
            title = question[:50]
            if len(question) > 50:
                title += "..."

            conversation = Conversation(
                session_id=session_id,
                title=title,
            )

            db.session.add(conversation)
            db.session.commit()

        previous_entity = conversation.last_entity
        previous_question = conversation.last_question

        # Simpan pertanyaan pengguna.
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=question,
        )

        db.session.add(user_message)
        db.session.commit()

    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Gagal menyiapkan atau menyimpan percakapan."
        )

        return jsonify({
            "error": "Gagal menyimpan percakapan ke database."
        }), 500

    # Jalankan retrieval dan generation melalui chatbot.
    try:
        app.logger.info("Memulai chatbot.ask()")

        result = chatbot.ask(
            question=question,
            previous_entity=previous_entity,
            previous_question=previous_question,
        )

        if not isinstance(result, dict):
            raise TypeError(
                "chatbot.ask() harus mengembalikan dictionary, "
                f"bukan {type(result).__name__}."
            )

        answer = result.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            answer = "Maaf, jawaban belum dapat dibuat saat ini."

        answer = answer.strip()
        found = bool(result.get("found", False))
        entity = result.get("entity")

        effective_question = result.get(
            "effective_question",
            question,
        )

        app.logger.info("chatbot.ask() berhasil.")

    except Exception:
        app.logger.exception(
            "Gagal memproses pertanyaan chatbot."
        )
        db.session.rollback()

        return jsonify({
            "error": (
                "Terjadi kesalahan saat memproses pertanyaan. "
                "Silakan coba kembali."
            )
        }), 500

    # Simpan jawaban dan konteks percakapan.
    try:
        conversation.last_entity = entity if found else None
        conversation.last_question = effective_question

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

    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Gagal menyimpan jawaban chatbot."
        )

        return jsonify({
            "error": (
                "Jawaban berhasil diproses, tetapi gagal "
                "disimpan ke database."
            )
        }), 500


# ============================================================
# CONVERSATION HISTORY API
# ============================================================

@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    session_id = current_session_id()

    try:
        conversations = (
            Conversation.query
            .filter_by(session_id=session_id)
            .order_by(Conversation.created_at.desc())
            .all()
        )

        return jsonify([
            conversation.to_dict()
            for conversation in conversations
        ])

    except Exception:
        app.logger.exception(
            "Gagal mengambil daftar percakapan."
        )

        return jsonify({
            "error": "Gagal mengambil riwayat percakapan."
        }), 500


@app.route(
    "/api/conversations/<int:conversation_id>/messages",
    methods=["GET"],
)
def get_conversation_messages(conversation_id):
    session_id = current_session_id()

    try:
        conversation = Conversation.query.filter_by(
            id=conversation_id,
            session_id=session_id,
        ).first()

        if conversation is None:
            return jsonify({
                "error": "Percakapan tidak ditemukan."
            }), 404

        return jsonify([
            message.to_dict()
            for message in conversation.messages
        ])

    except Exception:
        app.logger.exception(
            "Gagal mengambil pesan percakapan."
        )

        return jsonify({
            "error": "Gagal mengambil pesan percakapan."
        }), 500


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if current_user.is_authenticated:
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        admin = Admin.query.filter_by(email=email).first()

        if (
            admin is None
            or not check_password_hash(admin.password_hash, password)
        ):
            flash("Email atau password salah.", "error")
            return redirect(url_for("admin_login"))

        login_user(admin)

        flash("Login berhasil.", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_login.html")


@app.route("/admin/logout")
@login_required
def admin_logout():
    logout_user()
    flash("Kamu berhasil logout.", "success")

    return redirect(url_for("admin_login"))


@app.route("/adminpage")
@login_required
def admin_dashboard():
    return render_template("adminpage.html")


# ============================================================
# ADMIN DOCUMENT UPLOAD AND INGEST
# ============================================================

@app.route("/admin/upload-dokumen", methods=["POST"])
@login_required
def admin_upload_dokumen():
    uploaded_file = request.files.get("document")

    if uploaded_file is None or not uploaded_file.filename:
        flash("Pilih file PDF terlebih dahulu.", "warning")
        return redirect(url_for("admin_dashboard"))

    if not uploaded_file.filename.lower().endswith(".pdf"):
        flash("File harus berformat PDF.", "error")
        return redirect(url_for("admin_dashboard"))

    filename = secure_filename(uploaded_file.filename)

    if not filename:
        flash("Nama file tidak valid.", "error")
        return redirect(url_for("admin_dashboard"))

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename,
    )

    try:
        uploaded_file.save(file_path)

        result = ingest_document(file_path)

        if isinstance(result, dict) and result.get("success"):
            flash(
                result.get(
                    "message",
                    "Dokumen berhasil diproses dan Knowledge Base diperbarui.",
                ),
                "success",
            )
        else:
            if isinstance(result, dict):
                message = result.get(
                    "message",
                    "Gagal melakukan ingest dokumen.",
                )
            else:
                message = (
                    "Proses ingest tidak menghasilkan respons yang valid."
                )

            flash(message, "error")

    except Exception:
        app.logger.exception(
            "Gagal mengunggah atau melakukan ingest dokumen."
        )

        flash(
            "Terjadi kesalahan saat memproses dokumen. "
            "Periksa log aplikasi.",
            "error",
        )

    return redirect(url_for("admin_dashboard"))


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
