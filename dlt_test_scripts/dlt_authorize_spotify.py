"""
Authenticate spotify app using Auth code flow (not PKCE)
details: developer.spotify.com/documentation/web-api/concepts/authorization
"""

from uuid import uuid4
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
import json

import dlt

CALLBACK_OUTPUT = "auth_callback.json"
AUTH_SCOPES = ['user-read-recently-played']


class CallbackHandler(BaseHTTPRequestHandler):
    auth_code: str = None

    def do_GET(self):
        query = urlparse(self.path).query
        params = parse_qs(query)
        
        if "code" in params:
            self.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write("Authorization successful!".encode())
            with open(CALLBACK_OUTPUT, 'wt') as f:
                json.dump(params, f)
        elif "error" in params:
            self.send_response(400)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Error: {params['error']}".encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging


def start_server():
    return HTTPServer((
            dlt.config['sources.spotify.callback_addr'],
            dlt.config['sources.spotify.callback_port'],
        ), CallbackHandler)


def request_auth():
    params = {
        'response_type': 'code',
        'client_id': dlt.secrets['sources.spotify.key'],
        'scope': ' '.join(AUTH_SCOPES),
        'redirect_uri': "http://{}:{}".format(
            dlt.config['sources.spotify.callback_addr'],
            dlt.config['sources.spotify.callback_port'],
        ),
        'state': uuid4().hex
    }
    auth_url = "https://accounts.spotify.com/authorize?" + urlencode(params)
    print(f"Click here to authorize:")
    print(auth_url)


if __name__ == '__main__':
    server = start_server()
    request_auth()
    server.handle_request()
