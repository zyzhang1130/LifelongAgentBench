#!/usr/bin/env python
# run_env_server.py  (v2)

import argparse, pathlib, sys, os, requests, uvicorn
from src.utils import SingletonLogger  # lives under src/
from src.typings import LoggerConfig

REPO = pathlib.Path(__file__).resolve().parent
sys.path.append(str(REPO))
os.environ["PYTHONPATH"] = str(REPO)

from src.distributed_deployment_utils.server_side_controller.main import (
    ServerSideController,
)


def build_controller(port: int):
    logger = SingletonLogger.get_instance(
        LoggerConfig(
            level="INFO",
            log_file_path="./outputs/server_side_controller.log",
            logger_name="server_side_controller",
        )
    )
    ctrl = ServerSideController(logger)
    uvicorn.run(ctrl.app, host="0.0.0.0", port=port, log_level="debug")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/assignments/debugs/instance/db_bench.yaml",
        help="YAML config that start_server.py expects",
    )
    p.add_argument(
        "--ctrl-port",
        type=int,
        default=8003,
        help="Port for the controller’s FastAPI (default 8003)",
    )
    args = p.parse_args()

    # 1 – start the controller in a child process
    import multiprocessing, time

    proc = multiprocessing.Process(target=build_controller, args=(args.ctrl_port,))
    proc.start()
    time.sleep(5)  # give FastAPI a moment to come up

    # 2 – send the StartServerRequest
    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO / config_path).resolve()

    url = f"http://localhost:{args.ctrl_port}/start_server/"
    response = requests.post(url, json={"config_path": str(config_path)})
    print("Controller reply:", response.json())
