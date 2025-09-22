import network
import time
from umqtt.simple import MQTTClient
import ubinascii
import random
import machine

import M5
from M5 import Widgets

SERVER = "0.0.0.0"
CLIENT_ID = ubinascii.hexlify(machine.unique_id())
PUBLISH_TOPIC = b"sample"
SUBSCRIBE_TOPIC = b"IOT/location0/server"

def connect_to_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('Connecting to network...')
        wlan.connect(ssid, password)
        while not wlan.isconnected():
            pass
    print('Network config:', wlan.ifconfig())
    return wlan

def scan_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    networks = wlan.scan()
    networks.sort(key=lambda x: x[3], reverse=True)  # Sort by signal
    wifi_data = []
    for ssid, bssid, channel, RSSI, authmode, hidden in networks[:5]:
        bssid_str = ':'.join(f'{b:02x}' for b in bssid)
        wifi_data.append(f"BSSID: {bssid_str}, Signal: {RSSI} dBm")
    return "; ".join(wifi_data)

def subscribe_callback(topic, msg):
    print("Received:", (topic, msg))

if __name__ == '__main__':
    try:
        # M5 UI Setup
        M5.begin()
        Widgets.fillScreen(0x000000)
        exit_label = Widgets.Label("ENTRY DEVICE", 0, 100, 4.4, 0xffffff, 0x000000)  # White text

        wlan = connect_to_wifi('AgentP', '12341234')
        if SERVER == "0.0.0.0":
            SERVER = wlan.ifconfig()[2]
        c = MQTTClient(CLIENT_ID, SERVER)
        c.connect()
        print("Connected to %s" % SERVER)
        c.set_callback(subscribe_callback)
        c.subscribe(SUBSCRIBE_TOPIC)

        for k in range(1000):
            wifi_info = scan_wifi()
            prefix = 0
            message = f"{prefix} WiFi: {wifi_info}"

            c.check_msg()
            c.publish(PUBLISH_TOPIC, message.encode())
            print(f"Published Data: {message}")

            time.sleep(3)

        c.disconnect()

    except (Exception, KeyboardInterrupt) as e:
        try:
            from utility import print_error_msg
            print_error_msg(e)
        except ImportError:
            print("Unable to import utility module")
    finally:
        c.disconnect()
