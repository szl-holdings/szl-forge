import json, os, urllib.request
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "khipu")
def generate(messages):
    req = urllib.request.Request(
        OLLAMA_URL.rstrip("/") + "/api/chat",
        data=json.dumps({"model": MODEL, "messages": messages, "stream": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"]
