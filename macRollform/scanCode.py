import pywifi
from pywifi import const
import time

def scan_wifi():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]  # Use the first wireless interface
    iface.scan()
    time.sleep(1)  # Give it time to scan
    scan_results = iface.scan_results()

    networks = []
    for network in scan_results:
        ssid = network.ssid
        bssid = network.bssid
        signal = network.signal  # dBm (higher is stronger, closer to 0)
        networks.append((ssid, bssid, signal))

    # Sort by signal strength descending (closer to 0 is better)
    networks.sort(key=lambda x: x[2], reverse=True)
    return networks[:5]

def print_top_networks():
    while True:
        top_networks = scan_wifi()
        print(f"\nTop 5 WiFi Networks ({time.strftime('%H:%M:%S')}):")
        print("-" * 40)
        for ssid, bssid, signal in top_networks:
            print(f"| SSID: {ssid} | BSSID: {bssid} | Signal Strength: {signal} dBm |")
        time.sleep(3)

if __name__ == "__main__":
    print_top_networks()
