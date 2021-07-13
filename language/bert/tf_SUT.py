# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "TensorFlow", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
from squad_QSL import get_squad_QSL

class BERT_TF_SUT():
    def __init__(self, args):
        print("Loading TF model...")
        infer_config = tf.compat.v1.ConfigProto()
        infer_config.intra_op_parallelism_threads = int(os.environ['TF_INTRA_OP_PARALLELISM_THREADS']) \
                if 'TF_INTRA_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        print("intra op threads: {}".format(infer_config.intra_op_parallelism_threads))
        infer_config.inter_op_parallelism_threads = int(os.environ['TF_INTER_OP_PARALLELISM_THREADS']) \
                if 'TF_INTER_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        print("inter op threads: {}".format(infer_config.inter_op_parallelism_threads))
        infer_config.use_per_session_threads = 1
        infer_config.gpu_options.allow_growth = True
        self.batchsize = args.batchsize

        self.sess = tf.compat.v1.Session(config=infer_config)
        with gfile.FastGFile('build/data/bert_tf_v1_1_large_fp32_384_v2/model.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.max_examples)

    def run_with_each_sample(self, query_samples):
        for i in range(len(query_samples)):
            eval_features = self.qsl.get_features(query_samples[i].index)
            input_ids   = np.array([eval_features.input_ids])
            input_mask  = np.array([eval_features.input_mask])
            segment_ids = np.array([eval_features.segment_ids])
            feeds = {
                'input_ids:0':   input_ids,
                'input_mask:0':  input_mask,
                'segment_ids:0': segment_ids
            }
            result = self.sess.run(["logits:0"], feed_dict=feeds)

            logits = [float(x) for x in result[0].flat]
            response_array = array.array("B", np.array(logits).astype(np.float32).tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def run_one_batch(self, query_samples, cur_batch_size=1, base_index=0):
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        for i in range(cur_batch_size):
            idx = base_index + i
            eval_features = self.qsl.get_features(query_samples[idx].index)
            input_ids_list.append(eval_features.input_ids)
            input_mask_list.append(eval_features.input_mask)
            segment_ids_list.append(eval_features.segment_ids)
        feeds = {
            'input_ids:0': np.array(input_ids_list),
            'input_mask:0': np.array(input_mask_list),
            'segment_ids:0': np.array(segment_ids_list),
        }
        results = self.sess.run(["logits:0"], feed_dict=feeds)
        output = np.stack(results, axis=-1)
        out_list = np.split(output, cur_batch_size, axis = 0)
        for i, o in enumerate(out_list):
            idx = base_index + i
            response_array = array.array("B", np.array(o).astype(np.float32).tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[idx].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def issue_queries(self, query_samples):
        if self.batchsize > 1:
            num_samples = len(query_samples)
            num_batch = num_samples // self.batchsize
            remaining_batch = num_samples % self.batchsize
            for b in range(num_batch):
                base_index = b * self.batchsize
                self.run_one_batch(query_samples, self.batchsize, base_index)

            if remaining_batch > 0:
                base_index = num_batch * self.batchsize
                self.run_one_batch(query_samples, remaining_batch, base_index)
        else:
            self.run_with_each_sample(query_samples)

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_tf_sut(args):
    return BERT_TF_SUT(args)
