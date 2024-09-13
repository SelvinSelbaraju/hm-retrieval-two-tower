from typing import List, Optional, Dict
import tensorflow as tf
from pkg.modelling.models.tower import Tower
from pkg.schema.features import Feature
from pkg.schema.schema import Schema

class TwoTowerModel(tf.keras.Model):
    """
    Two Tower Model class

    Parameters
    ----------
    user_features: List[str]
        Features for the user tower
    item_features: List[str]
        Features for the item tower
    joint_embedding_size: int
        Joint embedding size which gets dot product
    user_tower_units: Optional[List[int]]
        Hidden units for the user tower
    item_tower_units: Optional[List[int]]
        Hidden units for the item tower
    """
    def __init__(
        self,
        user_features: List[Feature],
        item_features: List[Feature],
        joint_embedding_size: int,
        user_tower_units: Optional[List[int]] = None,
        item_tower_units: Optional[List[int]] = None
    ):
        super().__init__()
        self.user_tower = Tower(user_features, joint_embedding_size, user_tower_units)
        self.item_tower = Tower(item_features, joint_embedding_size, item_tower_units)
    
    def call(self, x: Dict[str, tf.Tensor]) -> tf.Tensor:
        users = self.user_tower(x)
        items = self.item_tower(x)
        return tf.matmul(users, items, transpose_b=True)


    @classmethod
    def create_from_schema(cls, schema: Schema) -> "TwoTowerModel":
        return TwoTowerModel(
            user_features=[f for f in schema.features if f.name in schema.model_config.user_features],
            item_features=[f for f in schema.features if f.name in schema.model_config.item_features],
            joint_embedding_size=schema.model_config.joint_embedding_size,
            user_tower_units=schema.model_config.user_tower_units,
            item_tower_units=schema.model_config.item_tower_units
        )

    

