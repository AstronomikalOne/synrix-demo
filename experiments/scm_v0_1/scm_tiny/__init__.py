"""SCM-Tiny — distillation scaffolding for route prediction (numpy-only baseline).

Trains a tiny softmax classifier on hashed packet features using ``RulesScmRouter``
as the teacher. Not a full learned controller; see spec § SCM-Tiny / SCM-Base.
"""

from .artifact import (
    ScmTinyArtifact,
    ScmTinyPredictor,
    train_classifiers,
)
from .baseline import (
    ROUTE_CLASSES,
    SoftmaxRouteClassifier,
    SoftmaxTemplateClassifier,
    route_to_index,
)
from .dataset import distillation_examples
from .features import PACKET_FEATURE_DIM, featurize_packet, featurize_packets
from .templates import QUERY_TEMPLATE_IDS, template_id_from_teacher, template_to_index

__all__ = [
    "ROUTE_CLASSES",
    "QUERY_TEMPLATE_IDS",
    "PACKET_FEATURE_DIM",
    "ScmTinyArtifact",
    "ScmTinyPredictor",
    "SoftmaxRouteClassifier",
    "SoftmaxTemplateClassifier",
    "distillation_examples",
    "featurize_packet",
    "featurize_packets",
    "route_to_index",
    "template_id_from_teacher",
    "template_to_index",
    "train_classifiers",
]
