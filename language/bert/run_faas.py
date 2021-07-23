import os
import sys
import json
import requests
sys.path.insert(0, os.getcwd())

import argparse
import mlperf_loadgen as lg
from squad_QSL import get_squad_QSL

# pylint: disable=missing-docstring

NANO_SEC = 1e9
MILLI_SEC = 1000


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--backend", choices=["tf","pytorch","onnxruntime","tf_estimator"], default="tf", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline", "Server", "MultiStream"], default="Offline", help="Scenario")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--quantized", action="store_true", help="use quantized model (only valid for onnxruntime backend)")
    parser.add_argument("--profile", action="store_true", help="enable profiling (only valid for onnxruntime backend)")
    parser.add_argument("--mlperf_conf", default="build/mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--max_examples", type=int, help="Maximum number of examples to consider (not limited by default)")
    parser.add_argument("--batchsize", type=int, help="maximum batch size")
    parser.add_argument("--output", default="output", help="test results")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="gateway url")
    parser.add_argument("--split", action="store_true", help="whether to split the queries")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    args = parser.parse_args()
    return args

scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}


class Runner:
    def __init__(self, url: str, batchsize: int, split: bool):
        self.url = url
        self.batchsize = batchsize
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
                'batch_size': self.batchsize,
            }
            self.issue_one_item(input)
        else:
            if len(query_samples) < self.max_batchsize:
                input = {
                    'qs': queries,
                    'batch_size': self.batchsize,
                }
                self.issue_one_item(input)
            else:
                for i in range(0, len(query_samples), self.batchsize):
                    sub_qs = queries[i:i+self.batchsize]
                    self.issue_one_item({
                        'qs': sub_qs,
                        'batch_size': self.batchsize,
                    })


def flush_queries():
    pass


def process_latencies(latencies_ns):
    pass


def load_query_samples(sample_list):
    pass


def unload_query_samples(sample_list):
    pass


def main():
    args = get_args()

    count_override = False
    count = args.count
    if count:
        count_override = True

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "bert", args.scenario)
    settings.FromConfig(args.user_conf, "bert", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "build/logs"

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    if count_override:
        settings.min_query_count = count
        settings.max_query_count = count

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = True

    full_url = args.url + '/function/faas_bert'
    runner = Runner(full_url, args.batchsize, args.split)

    def issue_queries(query_samples):
        runner.issue_queries(query_samples)

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = get_squad_QSL(args.max_examples)

    lg.StartTestWithLogSettings(sut, qsl.qsl, settings, log_settings)

    print("Destroying SUT...")
    lg.DestroySUT(sut)

    print("Destroying QSL...")
    lg.DestroyQSL(qsl.qsl)


if __name__ == '__main__':
    main()
