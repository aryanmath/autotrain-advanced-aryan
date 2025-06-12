from flask import Flask
from .routes.life_app_routes import life_app_bp

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(life_app_bp)
    
    return app
