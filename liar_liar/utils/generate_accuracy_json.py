import errno
import json
import os
from os import path
from liar_liar.models.base_models.sequential_model import get_all_models


def generate_accuracy_dict():
    all_models = get_all_models()
    accuracy_dict = {}
    for model in all_models:
        accuracy_dict[model.MODEL_NAME] = {}
        metrics = model.test()
        metrics_names = model.metrics_names
        for metric_val, metric_name in zip(metrics, metrics_names):
            accuracy_dict[model.MODEL_NAME][metric_name] = metric_val
    return accuracy_dict

def write_dict_to_json(dict):
    json_file_path = path.dirname(path.realpath(__file__))
    json_file_path = path.join(json_file_path,
                               path.pardir,
                               path.pardir,
                               "json_results",
                               "accuracy_results" + ".json")
    if not path.exists(path.dirname(json_file_path)):
        try:
            os.mkdir(path.dirname(json_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    acc_dict = generate_accuracy_dict()
    write_dict_to_json(acc_dict)
