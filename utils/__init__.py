"""
Utils Package
=============
Utility modules for RF-DETR detection system.

Modules:
    - id_helper: Bounding box ID generation
    - preprocessing: Frame preprocessing pipeline
    - label_io: Label file serialization
    - network_publish: MQTT and HTTP publishing
"""

from .id_helper import add_bbox_ids_to_detections
from .preprocessing import PreprocessedFrame, preprocess_frame
from .label_io import save_detections_to_txt, load_detections_from_txt
from .network_publish import (
    PublishResult,
    format_payload,
    configure_mqtt,
    publish_mqtt,
    publish_http
)

__all__ = [
    # id_helper
    'add_bbox_ids_to_detections',
    # preprocessing
    'PreprocessedFrame',
    'preprocess_frame',
    # label_io
    'save_detections_to_txt',
    'load_detections_from_txt',
    # network_publish
    'PublishResult',
    'format_payload',
    'configure_mqtt',
    'publish_mqtt',
    'publish_http',
]
