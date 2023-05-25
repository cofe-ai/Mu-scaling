import os
import evaluate

cur_path = os.path.dirname(__file__)


def my_evaluate_load(path, **kwargs):
    # 首先尝试从本地加载
    if os.path.isdir(path) or os.path.isfile(path):
        fun_eval = evaluate.load(path, **kwargs)
    else:
        try:
            local_path = os.path.abspath(os.path.join(cur_path, path))
            print(f'Load `{path}` From `{local_path}`')
            fun_eval = evaluate.load(local_path, **kwargs)
        except:
            print(f'Load `{path}` From `hub`')
            fun_eval = evaluate.load(path, **kwargs)

    return fun_eval
