# build.py
from flask_frozen import Freezer
from app import create_app  # Replace 'your_app' with your actual module name (e.g., 'app' or 'project')

# Create the Flask app instance (use your config, e.g., 'development' or 'production')
app = create_app()

# Optional: Configure relative URLs if your app uses them
app.config['FREEZER_RELATIVE_URLS'] = True

# Create Freezer instance
freezer = Freezer(app)

if __name__ == '__main__':
    # Generate static files into the 'build' directory
    freezer.freeze()
