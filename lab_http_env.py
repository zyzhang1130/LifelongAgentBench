# lab_http_env.py  -------------------------------------------------------
import requests, time
from src.typings.session import Session
from src.typings.request import TaskRequest


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
        session = Session(task_name="db_bench", sample_index=0)
        request = TaskRequest.Reset(session=session)
        response = requests.post(self.base + "/api/reset", json=request.model_dump())
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(
                f"Failed to decode JSON from server. Status code: {response.status_code}"
            )
            print("Server response:")
            print(response.text)
            raise e

    def step(self, action: str):
        """Send ONE action string; return dict with keys
        reward (0/1), observation, done, info."""
        return requests.post(
            self.base + "/api/interact", json={"action": action}
        ).json()


# ------------------------------------------------------------------------
