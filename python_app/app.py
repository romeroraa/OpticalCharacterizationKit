from flask import Flask, render_template
from apps.exp7 import exp7_blueprint
from apps.exp3 import exp3_blueprint
from apps.exp1_darkframe import exp1_darkframe_blueprint
from apps.exp1_linearity import exp1_linearity_blueprint
from apps.exp1_flat_field import exp1_flat_field_blueprint
from apps.exp5 import exp5_blueprint
from apps.exp1_mean_variance import exp1_mean_variance_blueprint
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

app.register_blueprint(exp7_blueprint, url_prefix="/exp7")
app.register_blueprint(exp3_blueprint, url_prefix="/exp3")
app.register_blueprint(exp1_darkframe_blueprint, url_prefix="/exp1_darkframe")
app.register_blueprint(exp1_linearity_blueprint, url_prefix="/exp1_linearity")
app.register_blueprint(exp1_flat_field_blueprint, url_prefix="/exp1_flat_field")
app.register_blueprint(exp5_blueprint, url_prefix="/exp5")
app.register_blueprint(exp1_mean_variance_blueprint, url_prefix="/exp1_mean_variance")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
