import os

# Demo-only shared secret for the local inference service.
#
# This is NOT a real credential: the API is meant to run on localhost for the
# thesis demo, and the key exists only so the endpoints are not wide open to
# anything else on the machine. Set FRAUD_API_KEY to override it.
API_KEY = os.getenv("FRAUD_API_KEY", "local-dev-demo-key")
API_KEY_NAME = "X-API-Key"  # header name
