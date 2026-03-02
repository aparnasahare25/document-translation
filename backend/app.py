"""
Flask application factory registers all blueprints.
"""

import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

    # Allow React dev‑server and same‑origin requests
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    from routes.pdf_routes import pdf_bp
    app.register_blueprint(pdf_bp, url_prefix="/api")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)