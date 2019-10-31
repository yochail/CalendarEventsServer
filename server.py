from flask import Flask, request, jsonify
from flask_cors import CORS

from events import event as ev
app = Flask(__name__)
CORS(app)
@app.route('/parse_text',methods=['POST'])
#@cross_origin(origins=['https://mail.google.com'])
def parse_text():
    json_obj = request.get_json()
    event = ev()
    event.load_event_data(json_obj)
    return jsonify(event.__dict__)

if __name__ == "__main__":
    app.run(ssl_context='adhoc')