"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import json
import logging
import os
import sys
import time
from timeit import default_timer as timer
import requests

import mlperf_loadgen as lg
import numpy as np

from settings import SUPPORTED_DATASETS, SUPPORTED_PROFILES, get_backend

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

# pylint: disable=missing-docstring
last_timeing = []


SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-format", choices=["NCHW", "NHWC"], help="data format")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--dataset-list", help="path to the dataset list")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    parser.add_argument("--max-batchsize", type=int, help="max batch size in a single inference")
    parser.add_argument("--output", default="output", help="test results")
    parser.add_argument("--inputs", help="model inputs")
    parser.add_argument("--outputs", help="model outputs")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--model-name", help="name of the mlperf model, ie. resnet50")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--cache", type=int, default=0, help="use cache")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")
    parser.add_argument("--debug", action="store_true", help="debug, turn traces on")
    parser.add_argument("--in_dtypes", help="input data types, uint8 or float32 (for tensorlow)")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--split", action="store_true", help="whether to split the query samples into smaller pieces")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", default="../../mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--samples-per-query", type=int, help="mlperf multi-stream sample per query")
    args = parser.parse_args()

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args


class Runner:
    def __init__(self, url: str, max_batchsize: int, split: bool):
        self.url = url
        self.max_batchsize = max_batchsize
        self.split = split

    def issue_one_item(self, query_input):
        json_input = json.dumps(query_input)
        ret = requests.post(self.url, data=json_input, verify=False)
        if ret.status_code == requests.codes.ok:
            ret = ret.json()
            if ret:
                response = [lg.QuerySampleResponse(r[0], r[1], r[2]) for r in ret['res']]
                lg.QuerySamplesComplete(response)

    def issue_queries(self, query_samples):
        queries = [{'idx': q.index, 'id': q.id} for q in query_samples]
        if not self.split:
            input = {
                'qs': queries,
                'batch_size': self.max_batchsize,
            }
            self.issue_one_item(input)
        else:
            if len(query_samples) < self.max_batchsize:
                input = {
                    'qs': queries,
                    'batch_size': self.max_batchsize,
                }
                self.issue_one_item(input)
            else:
                for i in range(0, len(query_samples), self.max_batchsize):
                    sub_qs = queries[i:i+self.max_batchsize]
                    self.issue_one_item({
                        'qs': sub_qs,
                        'batch_size': self.max_batchsize,
                    })


def flush_queries():
    pass


def process_latencies(latencies_ns):
    # called by loadgen to show us the recorded latencies
    global last_timeing
    last_timeing = [t / NANO_SEC for t in latencies_ns]


def load_query_samples(sample_list):
    pass

def unload_query_samples(sample_list):
    pass


def main():
    args = get_args()
    log.info(args)

    # find backend
    backend = get_backend(args.backend)

    # override image format if given
    image_format = args.data_format if args.data_format else backend.image_format()

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    count_override = False
    count = args.count
    if count:
        count_override = True

    # dataset to use
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = wanted_dataset(data_path=args.dataset_path,
                        image_list=args.dataset_list,
                        name=args.dataset,
                        image_format=image_format,
                        pre_process=pre_proc,
                        use_cache=args.cache,
                        count=count, **kwargs)

    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    count = ds.get_item_count()
    scenario = SCENARIO_MAP[args.scenario]
    full_url = args.url + '/function/faas_classification_detection'
    runner = Runner(full_url, args.max_batchsize, args.split)

    def issue_queries(query_samples):
        start = timer()
        runner.issue_queries(query_samples)
        end = timer()
        print(f"sample/s: {len(query_samples)/(end-start)}")

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir
    log_output_settings.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.enable_trace = args.debug
    log_settings.log_output = log_output_settings

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, args.model_name, args.scenario)
    settings.FromConfig(user_conf, args.model_name, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    if count_override:
        settings.min_query_count = count
        settings.max_query_count = count

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query
    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_target_latency_ns = int(args.max_latency * NANO_SEC)

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(count, min(count, 500), load_query_samples, unload_query_samples)

    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    print("Done!")

    print("Destroying SUT...")
    lg.DestroyQSL(qsl)
    print("Destroying QSL...")
    lg.DestroySUT(sut)


if __name__ == '__main__':
    main()
