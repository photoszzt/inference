import os
import pickle
from transformers import BertTokenizer
from create_squad_data import read_squad_examples, convert_examples_to_features

max_seq_length = 384
max_query_length = 64
doc_stride = 128


def main():
    cache_path = 'eval_features.pickle'
    eval_features = []
    # Load features if cached, convert from examples otherwise.
    if os.path.exists(cache_path):
        print("Loading cached features from '%s'..." % cache_path)
        with open(cache_path, 'rb') as cache_file:
            eval_features = pickle.load(cache_file)
    else:
        print("No cached features at '%s'... converting from examples..." % cache_path)
        print("Creating tokenizer...")
        tokenizer = BertTokenizer("build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt")
        print("Reading examples...")
        eval_examples = read_squad_examples(input_file="build/data/dev-v1.1.json",
            is_training=False, version_2_with_negative=False)
        print("Converting examples to features...")
        def append_feature(feature):
            eval_features.append(feature)
        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            output_fn=append_feature,
            verbose_logging=False)
        print("Caching features at '%s'..." % cache_path)
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(eval_features, cache_file)


if __name__ == '__main__':
    main()
