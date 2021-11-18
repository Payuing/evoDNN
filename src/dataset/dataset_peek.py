from typing import Callable


def data_peek(dataset_name: str, dataloader: Callable) -> None:
    features, target = dataloader()
    print(f"{dataset_name} dataset")
    print(f"Features data type: {type(features)}")
    print(f"First 5 rows in features: \n{features[:5]}")
    print(f"Target data type: type(target)")
    print(f"First 5 rows in target: {target[:5]}")
