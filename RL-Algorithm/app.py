from flask import Flask, render_template, jsonify
import json
from reinforcementLearning import load_q_table

app = Flask(__name__)

# Environment settings
grid_size = (10, 10)
start = (0, 0)
goal = (2, 8)
q_table = load_q_table('q_table.json')


@app.route('/')
def index():
    return render_template('QTable.html')


@app.route('/q_table')
def get_q_table():
    return jsonify(q_table)


if __name__ == '__main__':
    app.run(debug=True)
