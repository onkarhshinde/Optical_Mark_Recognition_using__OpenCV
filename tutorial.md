# Steps to follow


## Create Broker
- Go to mosquitto folder and update ```mosquitto.conf``` file to add ```listener 1883 0.0.0.0``` and ```allow_anonymous true``` and save.
- Now run the command ```mosquitto.exe -v -c mosquitto.conf``` or ```.\mosquitto.exe -v -c mosquitto.conf``` to activate the broker service OR go to "service" from start menu and 'start' the Mosquitto Service from UI. File locations is "C:\Program Files\mosquitto".
- Open the MQTT Explorer UI and create a new connection with:
    - Name: Custom
    - Protocol : mqtt://
    - Host: IPv4 of local hotspot
    - Port: 1883

---


## Publish/ Subscribe

### Publish
In the mosquitto folder open terminal and type:
```
mosquitto_pub -t topic_name/sub_topic_name(optional) -h broker_IPV4(ex: 192.168.137.1) -m "your_message"
```


### Subscribe


In the mosquitto folder open terminal and type:
```
mosquitto_sub -t topic_name/sub_topic_name -h 192.168.137.1
```

use ```#``` for subscribing 'all'.

Ex:
```
mosquitto_sub -t topic_name/# -h 192.168.137.1
```