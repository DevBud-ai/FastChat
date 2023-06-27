"""
A model worker executes the model based on vLLM.

This is an experimental feature and will be documented soon. Please stay tuned!

Install vLLM (``pip install vllm'') first. Then, assuming the controller is live:
1. python3 -m fastchat.serve.vllm_worker --model-path path_to_vicuna

launch Gradio:
2. python3 -m fastchat.serve.vllm_worker --concurrency-count 10000
"""
import threading

import argparse
import asyncio
import json
import time
import uuid

import requests
import torch
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL
from fastchat.utils import build_logger, pretty_print_semaphore
from fastchat.model.model_adapter import (
    get_conversation_template,
)
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


GB = 1 << 30
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0
seed = torch.cuda.current_device()


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class VLLMWorker:
    def __init__(
            self,
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            model_name):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        # if model_path.endswith("/"):
        #     model_path = model_path[:-1]
        self.model_name = model_name
        logger.info(f"Loading the model {self.model_name} on worker {worker_id}, worker type: vLLM worker...")

        self.conv = get_conversation_template(model_path)
        self.context_len = 2048

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()


    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {[self.model_name]}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}. "
            f"worker_id: {worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def count_token(self, params):
        prompt = params["prompt"]
        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret
    
    def get_conv_template(self):
        return {"conv": self.conv}

    async def generate_stream(self, params):
        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        echo = params.get("echo", True)

        # Handle stop_str
        if stop_str is None:
            stop_str = []

        # TODO(Hao): handle stop token IDs
        # stop_token_ids = params.get("stop_token_ids", None) or []
        # max_src_len = self.context_len - max_new_tokens - 8
        # input_ids = input_ids[-max_src_len:]

        input_echo_len = len(context) # TODO: change to token length

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=False,
            stop=stop_str,
            max_tokens=max_new_tokens
        )
        results_generator = engine.generate(context, sampling_params, request_id)

        token_count = 1
        async for request_output in results_generator:
            prompt = request_output.prompt
            if echo:
                text_outputs = [
                    prompt + output.text
                    for output in request_output.outputs
                ]
            else:
                text_outputs = [output.text for output in request_output.outputs]
            text_outputs = " ".join(text_outputs)
            ret = {"text": text_outputs, "error_code": 0}
            ret['usage'] = {
                "prompt_tokens": input_echo_len,
                "completion_tokens": token_count,
                "total_tokens": input_echo_len + token_count,
            }
            ret['finish_reason'] = 'stop'
            token_count += 1 
            yield (json.dumps(ret) + "\0").encode("utf-8")


app = FastAPI()
model_semaphore = None


def release_model_semaphore():
    model_semaphore.release()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    request_id = random_uuid()
    params = await request.json()
    params["request_id"] = request_id

    async def abort_request() -> None:
        await engine.abort(request_id)

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    background_tasks.add_task(abort_request)
    return StreamingResponse(
        worker.generate_stream(params), background=background_tasks
    )


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()

@app.post("/count_token")
async def count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)

@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()

@app.post("/model_details")
async def model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--model-path", type=str, default=""
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.download_dir = args.model_path
    # if args.model_names:
    #     args.model = args.model_names
    
    args.model = args.model_path
    worker = VLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_path,
        args.model_name
    )
    engine_args = AsyncEngineArgs.from_cli_args(args)
    print(engine_args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
