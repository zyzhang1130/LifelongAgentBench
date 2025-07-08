# lab_http_env.py  -------------------------------------------------------
import requests, time


class LABHTTPEnv:
    """Simple, stateless HTTP wrapper for LifelongAgentBench."""

    def __init__(self, port: int = 5000):
        self.base = f"http://localhost:{port}"
        # health‑check
        for _ in range(30):
            try:
                if requests.post(self.base + "/api/ping", timeout=1).ok:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            raise RuntimeError("LAB server not responding")

    def reset(self):
        return requests.post(self.base + "/api/reset").json()

    def step(self, action: str):
        """Send ONE action string; return dict with keys
        reward (0/1), observation, done, info."""
        return requests.post(
            self.base + "/api/interact", json={"action": action}
        ).json()


# ------------------------------------------------------------------------
