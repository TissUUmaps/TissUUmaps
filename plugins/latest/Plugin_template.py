import logging
import time

from flask import abort, make_response


class Plugin:
    def __init__(self, app):
        self.app = app

    def server_demo(self, jsonParam):
        if not jsonParam:
            logging.error("No arguments, aborting.")
            abort(500)
        time.sleep(1)
        resp = make_response("The server received the message: " + jsonParam["message"])
        return resp
