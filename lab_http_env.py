# lab_http_env.py  -------------------------------------------------------
import requests, time
import copy  # add for snapshot functionality
from src.typings.session import Session
from src.typings.request import TaskRequest


class LABHTTPEnv:
    """Simple, stateless HTTP wrapper for LifelongAgentBench."""

    def __init__(self, port: int = 5000):
        self.base = f"http://localhost:{port}"
        self.current_session = None
        self.current_sample_index = 0  # Track current task index
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
        session = Session(task_name="db_bench", sample_index=self.current_sample_index)
        request = TaskRequest.Reset(session=session)
        response = requests.post(
            self.base + "/api/reset", json=request.model_dump(), timeout=300
        )
        try:
            result = response.json()
            # Check if server returned an error
            if "detail" in result:
                raise RuntimeError(f"Server reset failed: {result['detail']}")
            # Store the current session state
            self.current_session = result["session"]
            # Increment sample index for next reset
            self.current_sample_index += 1
            print(
                f"Reset to task {self.current_sample_index - 1}, next will be {self.current_sample_index}"
            )
            return result
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
        if self.current_session is None:
            raise RuntimeError("Must call reset() before step()")

        # Update session with the agent's action
        self.current_session["chat_history"]["value"].append(
            {"role": "agent", "content": action}
        )

        # Create the interact request
        request = TaskRequest.Interact(session=Session(**self.current_session))
        response = requests.post(
            self.base + "/api/interact",
            json=request.model_dump(),
            timeout=300,  # 5 minutes
        )

        try:
            result = response.json()
            # Update current session state
            self.current_session = result["session"]

            # Convert to gym-style response
            return self._convert_to_gym_response(result)
        except requests.exceptions.JSONDecodeError as e:
            print(
                f"Failed to decode JSON from server. Status code: {response.status_code}"
            )
            print("Server response:")
            print(response.text)
            raise e

    def complete(self):
        """Complete the current task - must be called before starting a new task."""
        if self.current_session is None:
            raise RuntimeError("No active session to complete")

        # Create the complete request
        request = TaskRequest.Complete(session=Session(**self.current_session))
        response = requests.post(
            self.base + "/api/complete", json=request.model_dump(), timeout=300
        )

        try:
            result = response.json()
            # Check if server returned an error
            if "detail" in result:
                raise RuntimeError(f"Server complete failed: {result['detail']}")
            return result
        except requests.exceptions.JSONDecodeError as e:
            print(
                f"Failed to decode JSON from server. Status code: {response.status_code}"
            )
            print("Server response:")
            print(response.text)
            raise e

    def _convert_to_gym_response(self, server_response):
        """Convert server response to gym-style format expected by training script."""
        session = server_response["session"]

        # Determine if task is done
        done = session["sample_status"] != "running"

        # Calculate reward (1 if correct, 0 if incorrect, 0 if still running)
        if done:
            outcome = session["evaluation_record"]["outcome"]
            reward = 1 if outcome == "correct" else 0
        else:
            reward = 0

        # Extract observation (last user message in chat history)
        observation = ""
        chat_history = session["chat_history"]["value"]
        for msg in reversed(chat_history):
            if msg["role"] == "user":
                observation = msg["content"]
                break

        return {
            "reward": reward,
            "observation": observation,
            "done": done,
            "info": {"session": session},
        }

    def snapshot(self):
        """Deep copy the current in‑memory session so we can test candidates
        from the exact same state without committing."""
        if self.current_session is None:
            raise RuntimeError("Must call reset() before snapshot()")
        # Return in the same shape the trainer expects (with a top-level "session")
        return {"session": copy.deepcopy(self.current_session)}

    def release(self):
        """Release the current sample so server can reset safely."""
        r = requests.post(self.base + "/api/release", timeout=60)
        if not r.ok:
            raise RuntimeError(f"Server release failed: {r.text}")
        return True

    def simulate(self, action: str, snapshot: dict):
        """Evaluate an action from the provided snapshot without modifying
        the live session tracked in self.current_session."""
        # Prepare an isolated session copy
        local = copy.deepcopy(snapshot)
        local["chat_history"]["value"].append({"role": "agent", "content": action})

        request = TaskRequest.Interact(session=Session(**local))
        response = requests.post(
            self.base + "/api/interact", json=request.model_dump(), timeout=300
        )
        try:
            result = response.json()
            # Do NOT update self.current_session here
            return self._convert_to_gym_response(result)
        except requests.exceptions.JSONDecodeError as e:
            print(
                f"Failed to decode JSON from server. Status code: {response.status_code}"
            )
            print("Server response:")
            print(response.text)
            raise e


# ------------------------------------------------------------------------
