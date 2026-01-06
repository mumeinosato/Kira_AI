class VtubeStudioClient:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.authenticated = False

    def connect(self):
        self.ws = websocket.WebSocketApp(
            "ws://localhost:8001",
            on_message=self.on_message,
            on_open=self.on_open,
            on_error=self.on_error,
            on_close=self.on_close
        )
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def on_message(self, ws, message):
        print(f"Received message: {message}")
        data = json.loads(message)
        if data.get("messageType") == "AuthenticationTokenResponse":
            token = data["data"]["authenticationToken"]
            self.authenticate(token)
        elif data.get("messageType") == "AuthenticationResponse":
            if data["data"].get("authenticated"):
                print("Authentication successful!")
                self.authenticated = True
            else:
                print("Authentication failed.")

    def on_open(self, ws):
        print("WebSocket connection opened.")
        self.connected = True
        self.request_authentication_token()

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed.")
        self.connected = False

    def request_authentication_token(self):
        if not self.connected:
            print("WebSocket is not connected. Cannot send authentication token request.")
            return

        message = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "authToken",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": "YourPluginName",
                "pluginDeveloper": "YourName"
            }
        }
        self.ws.send(json.dumps(message))

    def authenticate(self, token):
        message = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "authRequest",
            "messageType": "AuthenticationRequest",
            "data": {
                "authenticationToken": token,
                "pluginName": "YourPluginName",
                "pluginDeveloper": "YourName"
            }
        }
        self.ws.send(json.dumps(message))

    def send_lip_sync(self,phonemes_with_timing):
        if self.connected and self.ws:
            message = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "inject-open",
                "messageType": "InjectParameterDataRequest",
                "data": {
                    "faceFound": False,
                    "mode": "set",
                    "parameterValues": [
                        {
                            "id": "MouthOpen",
                            "value": phonemes_with_timing.get("smile", 0)
                        }
                    ]
                }
            }
            self.ws.send(json.dumps(message))