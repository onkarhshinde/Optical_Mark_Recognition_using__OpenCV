
## Frist part: sending recieving done
- first character = 1 for exit
- first character = 0 for entry
- buffer time = 3 minutes


## Second part: getting registration






## Problems
- True physical mac address and publically shown mac addresses(randomised) are different
- Solution: ask students to open their hotspot and then enter their mac address.
- If you’re building an attendance system that scans for hotspot MACs, you need the AP MAC, not the STA MAC.







## STEPS TO OPERATE:


- Set up mosquitto server.
- Set up MySQL server.
- run ```python app.py``` in macRollform
- Open scanCode.py from macRollform
- From SSID, copy bssid of a given student.
- In the webpage, paste that SSID and add roll No.
- Run script on M5Stack.
- Run reciveing and uploading script `1forall.ipynb` present in the `Mosquitto` folder.




## Plots

1. Past 7 days : SIDE BY SIDE BAR CHART
    - num of entry 
    - num of exit

2. Previous Day hourly : LINE CHART
    - num of entry 
    - num of exit

3. Past 30 days average time spent data (category wise) : BAR CHART

    - year
    - gender
    - stream
    - branch
    - hostel


4. Nmber of entries (categrorical): PIE CHART


5. Hour wise number of entries past 30 days : line chart

6. One axis week , other axis hours to show heatmap

7. last  5 entries.