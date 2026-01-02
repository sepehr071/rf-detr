"""
Network Publishing Module
=========================
MQTT and HTTP publishing for detection results.

This module consolidates the network publishing logic that was duplicated
in main.py between run_detection_loop() and run_image_inference().
"""

import os
import json
import time
from typing import List, Optional
from dataclasses import dataclass

from config import (
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_DEFAULT_TOPIC,
    HTTP_ENDPOINT,
    HTTP_TIMEOUT,
    PAYLOAD_STORE_ID,
    PAYLOAD_DEVICE_ID,
    PAYLOAD_LEVEL,
    MQTT_LOGS_DIR
)


@dataclass
class PublishResult:
    """Result of a publish operation."""
    success: bool
    message: str
    status_code: Optional[int] = None


def format_payload(
    positions: List[dict],
    store_id: int = PAYLOAD_STORE_ID,
    device_id: int = PAYLOAD_DEVICE_ID,
    level: int = PAYLOAD_LEVEL
) -> dict:
    """
    Transform positioning data to payload format.

    This function was previously format_mqtt_payload() in main.py (lines 114-159).

    Args:
        positions: List of position dictionaries from calculate_positions()
        store_id: Store identifier
        device_id: Device identifier
        level: Shelf level

    Returns:
        Dict with store_id, device_id, products, and other_products
    """
    products = []
    other_products = []

    for pos in positions:
        if 'shelf_position' not in pos:
            continue  # Skip error entries

        item = {
            "class_id": pos["class_id"],
            "level": level,
            "facing": pos.get("rotation", 0),
            "position": {
                "x": pos["shelf_position"][0],
                "y": pos["shelf_position"][1]
            }
        }

        if pos["class_id"] == -1:
            other_products.append(item)
        else:
            products.append(item)

    return {
        "store_id": store_id,
        "device_id": device_id,
        "products": products,
        "other_products": other_products
    }


def configure_mqtt(
    broker_host: str = MQTT_BROKER_HOST,
    broker_port: int = MQTT_BROKER_PORT,
    topic: str = MQTT_DEFAULT_TOPIC
) -> None:
    """
    Configure MQTT handler with broker settings.

    Args:
        broker_host: MQTT broker hostname
        broker_port: MQTT broker port
        topic: Default MQTT topic
    """
    import mqtt_handler
    mqtt_handler.configure(
        broker_host=broker_host,
        broker_port=broker_port,
        default_topic=topic
    )


def publish_mqtt(
    positions: List[dict],
    save_log: bool = True,
    log_dir: str = MQTT_LOGS_DIR,
    log_name: Optional[str] = None
) -> PublishResult:
    """
    Publish positions via MQTT.

    This consolidates the MQTT publishing logic from main.py
    (detection loop lines 420-436, image inference lines 722-746).

    Args:
        positions: List of position dictionaries
        save_log: Whether to save payload to log file
        log_dir: Directory for log files
        log_name: Optional custom log filename (default: mqtt_{timestamp}.txt)

    Returns:
        PublishResult with success status
    """
    import mqtt_handler

    if not positions:
        return PublishResult(success=False, message="No positions to publish")

    payload = format_payload(positions)
    payload_json = json.dumps(payload, indent=2)

    # Save to log file if requested
    if save_log:
        os.makedirs(log_dir, exist_ok=True)
        if log_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_name = f"mqtt_{timestamp}.txt"
        log_path = os.path.join(log_dir, log_name)
        with open(log_path, 'w') as f:
            f.write(payload_json)

    # Send via MQTT
    rc = mqtt_handler.send(payload_json)
    if rc == 0:
        return PublishResult(
            success=True,
            message=f"Sent {len(positions)} positions",
            status_code=rc
        )
    else:
        return PublishResult(
            success=False,
            message=f"Failed to send (rc={rc})",
            status_code=rc
        )


def publish_http(
    positions: List[dict],
    endpoint: str = HTTP_ENDPOINT,
    timeout: int = HTTP_TIMEOUT
) -> PublishResult:
    """
    Publish positions via HTTP POST.

    This consolidates the HTTP publishing logic from main.py
    (detection loop lines 439-453, image inference lines 749-763).

    Args:
        positions: List of position dictionaries
        endpoint: HTTP endpoint URL
        timeout: Request timeout in seconds

    Returns:
        PublishResult with success status
    """
    import requests

    if not positions:
        return PublishResult(success=False, message="No positions to publish")

    payload = format_payload(positions)

    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        if response.status_code == 200:
            return PublishResult(
                success=True,
                message=f"Sent {len(positions)} positions",
                status_code=response.status_code
            )
        else:
            return PublishResult(
                success=False,
                message=f"Server returned status={response.status_code}",
                status_code=response.status_code
            )
    except requests.exceptions.RequestException as e:
        return PublishResult(
            success=False,
            message=f"Request failed: {e}"
        )
