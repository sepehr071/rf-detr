"""
mqtt_handler.py

Simple MQTT publish helper.

Usage from another script:

    import mqtt_handler as mqtt

    mqtt.send("Hello world")
    mqtt.send("With custom QoS", qos=2)
    mqtt.send("On another topic", topic="other/topic")
"""

import threading
import paho.mqtt.client as mqtt

# -------------------------------------------------------------------
# Default configuration (change these to fit your setup)
# -------------------------------------------------------------------
BROKER_HOST = "localhost"   # or your broker IP, e.g. "192.168.1.50"
BROKER_PORT = 1883
DEFAULT_TOPIC = "test/topic"
DEFAULT_QOS = 0             # 0, 1, or 2
CLIENT_ID = "python-lib-publisher"

# Internal globals
_client = None
_client_lock = threading.Lock()


# -------------------------------------------------------------------
# Optional: simple callbacks for logging/debugging
# -------------------------------------------------------------------
def _on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected to {BROKER_HOST}:{BROKER_PORT} with rc={rc}")


def _on_disconnect(client, userdata, rc):
    print(f"[MQTT] Disconnected with rc={rc}")


# -------------------------------------------------------------------
# Internal helper to lazily create and connect the client
# -------------------------------------------------------------------
def _get_client() -> mqtt.Client:
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        # Double-check inside lock
        if _client is not None:
            return _client

        client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
        client.on_connect = _on_connect
        client.on_disconnect = _on_disconnect

        # Blocking connect; raises an exception if it fails
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

        # Run network handling in a background thread
        client.loop_start()

        _client = client
        return _client


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def configure(
    broker_host: str | None = None,
    broker_port: int | None = None,
    default_topic: str | None = None,
    default_qos: int | None = None,
    client_id: str | None = None,
) -> None:
    """
    Optionally configure the module BEFORE the first send() call.

    Example:
        import mqtt_handler as mqtt
        mqtt.configure(broker_host="192.168.1.50", default_topic="my/topic")
    """
    global BROKER_HOST, BROKER_PORT, DEFAULT_TOPIC, DEFAULT_QOS, CLIENT_ID

    if _client is not None:
        raise RuntimeError("Cannot reconfigure after client has been created. "
                           "Call configure() before send().")

    if broker_host is not None:
        BROKER_HOST = broker_host
    if broker_port is not None:
        BROKER_PORT = broker_port
    if default_topic is not None:
        DEFAULT_TOPIC = default_topic
    if default_qos is not None:
        DEFAULT_QOS = default_qos
    if client_id is not None:
        CLIENT_ID = client_id


def send(
    payload: str,
    topic: str | None = None,
    qos: int | None = None,
    retain: bool = False,
) -> int:
    """
    Publish a message to the MQTT broker.

    Minimal usage:
        send("some text")

    Parameters:
        payload: The message payload (string).
        topic:   Topic to publish to. Defaults to DEFAULT_TOPIC.
        qos:     QoS level (0, 1, or 2). Defaults to DEFAULT_QOS.
        retain:  MQTT retain flag (False by default).

    Returns:
        The paho-mqtt result code (0 means success).
    """
    client = _get_client()

    publish_topic = topic if topic is not None else DEFAULT_TOPIC
    publish_qos = qos if qos is not None else DEFAULT_QOS

    result = client.publish(
        publish_topic,
        payload=payload,
        qos=publish_qos,
        retain=retain,
    )

    # Wait for the publish to complete (important for QoS 1/2)
    result.wait_for_publish()

    return result.rc
