"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:split_params"],
                outputs=[
                    "X_train",
                    "X_valid",
                    "X_test",
                    "y_train",
                    "y_valid",
                    "y_test",
                ],
                name="Split_Data_Node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "X_valid", "y_train", "y_valid"],
                outputs="Model",
                name="Train_Model_Node",
            ),
            node(
                func=evaluate_model,
                inputs=["Model", "X_test", "y_test"],
                outputs=None,
                name="Evaluate_Model_Node",
            ),
        ]
    )
