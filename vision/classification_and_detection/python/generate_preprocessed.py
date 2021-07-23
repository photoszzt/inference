import argparse
from settings import SUPPORTED_DATASETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--data-format", choices=["NCHW", "NHWC"], help="data format")
    args = parser.parse_args()
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    DS = wanted_dataset(data_path=args.dataset_path,
                        image_list=None,
                        name=args.dataset,
                        image_format=args.data_format,
                        pre_process=pre_proc,
                        use_cache=0,
                        count=None, **kwargs)


if __name__ == '__main__':
    main()