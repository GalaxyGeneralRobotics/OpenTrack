from collections import defaultdict
from collections.abc import Callable

# Nested dict: registry[task][category] -> object
_REGISTRY = defaultdict(dict)
_CATEGORIES = [
    "tracking_train_env_class",
    "tracking_play_env_class",
    "tracking_dagger_play_env_class",
    "tracking_play_command_class",
    "tracking_config",
    "tracking_dagger_config",
]


def _check_set_task(task: str):
    if task in _REGISTRY:
        raise ValueError(f"{task} is already registered, please use a different name")


def _check_get_task(task: str):
    if "_" in task:
        raise ValueError(f"task name should not contain '_' (underscores), got {task}")

    if task not in _REGISTRY:
        raise ValueError(f"{task} is not registered, available tasks are: {list(_REGISTRY.keys())}")


def _check_set_category(task: str, category: str):
    if category not in _CATEGORIES:
        raise ValueError(f"{category} is not a valid category, available categories are: {_CATEGORIES}")
    if category in _REGISTRY[task]:
        raise ValueError(f"{category} is already registered under task {task}, please use a different name")


def _check_get_category(task: str, category: str):
    _check_get_task(task)
    if category not in _REGISTRY[task]:
        raise ValueError(
            f"{category} is not registered in task {task}, available categories are {list(_REGISTRY[task].keys())}"
        )


def register(task: str, category: str):
    _check_set_category(task, category)

    def decorator(obj):
        _REGISTRY[task][category] = obj
        return obj

    return decorator


def get(task: str, category: str, call: bool = False):
    _check_get_category(task, category)

    if category not in _REGISTRY[task]:
        raise KeyError(f"{task} not registered under category '{category}'")
    obj = _REGISTRY[task][category]
    if call and callable(obj):
        return obj()
    return obj


def list_task():
    return list(_REGISTRY.keys())


def list_category(task: str):
    _check_get_task(task)
    return list(_REGISTRY[task].keys())
