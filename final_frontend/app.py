



from flask import Flask, render_template, jsonify
import mysql.connector

app = Flask(__name__)

# 🔗 DB Connection
def get_db_connection():
    return mysql.connector.connect(
        host='192.168.137.1',
        user='root',
        password='12341234',
        database='trial3'
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def fetch_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 1. Entries and exits over past 7 days
    cursor.execute("""
        SELECT DATE(entry_datetime) AS date, COUNT(*) AS num_entries
        FROM exit_entry_logs
        WHERE entry_datetime >= CURDATE() - INTERVAL 7 DAY
        GROUP BY DATE(entry_datetime)
        ORDER BY date;
    """)
    entries_7days = cursor.fetchall()

    cursor.execute("""
        SELECT DATE(exit_datetime) AS date, COUNT(*) AS num_exits
        FROM exit_entry_logs
        WHERE exit_datetime >= CURDATE() - INTERVAL 7 DAY
        GROUP BY DATE(exit_datetime)
        ORDER BY date;
    """)
    exits_7days = cursor.fetchall()

    # 2. Previous day hourly entries & exits
    cursor.execute("""
        SELECT HOUR(entry_datetime) AS hr, COUNT(*) AS num_entries
        FROM exit_entry_logs
        WHERE DATE(entry_datetime) = CURDATE() - INTERVAL 1 DAY
        GROUP BY hr
        ORDER BY hr;
    """)
    prevday_entries = cursor.fetchall()

    cursor.execute("""
        SELECT HOUR(exit_datetime) AS hr, COUNT(*) AS num_exits
        FROM exit_entry_logs
        WHERE DATE(exit_datetime) = CURDATE() - INTERVAL 1 DAY
        GROUP BY hr
        ORDER BY hr;
    """)
    prevday_exits = cursor.fetchall()

    # 3. Average time spent (per category)
    categories = ['year', 'gender', 'stream', 'branch', 'hostel']
    avg_time_by_category = {}
    entry_count_by_category = {}
    hourwise_category_entries = {}

    for cat in categories:
        # Avg time
        query_avg = f"""
            SELECT sd.{cat} AS category, 
                   ROUND(AVG(TIMESTAMPDIFF(MINUTE, eel.exit_datetime, eel.entry_datetime)),2) AS avg_duration
            FROM exit_entry_logs eel
            JOIN mac_roll mr ON eel.mac = mr.mac
            JOIN student_data sd ON mr.rollno = sd.rollno
            WHERE eel.entry_datetime >= CURDATE() - INTERVAL 30 DAY
            GROUP BY sd.{cat}
            ORDER BY sd.{cat};
        """
        cursor.execute(query_avg)
        avg_time_by_category[cat] = cursor.fetchall()

        # Entry count by category
        query_count = f"""
            SELECT sd.{cat} AS category, COUNT(*) AS count
            FROM exit_entry_logs eel
            JOIN mac_roll mr ON eel.mac = mr.mac
            JOIN student_data sd ON mr.rollno = sd.rollno
            WHERE eel.entry_datetime >= CURDATE() - INTERVAL 30 DAY
            GROUP BY sd.{cat}
            ORDER BY sd.{cat};
        """
        cursor.execute(query_count)
        entry_count_by_category[cat] = cursor.fetchall()

        # Hour-wise per category
        query_hourwise = f"""
            SELECT HOUR(eel.entry_datetime) AS hr, sd.{cat} AS category, COUNT(*) AS count
            FROM exit_entry_logs eel
            JOIN mac_roll mr ON eel.mac = mr.mac
            JOIN student_data sd ON mr.rollno = sd.rollno
            WHERE eel.entry_datetime >= CURDATE() - INTERVAL 30 DAY
            GROUP BY hr, sd.{cat}
            ORDER BY hr, sd.{cat};
        """
        cursor.execute(query_hourwise)
        hourwise_category_entries[cat] = cursor.fetchall()

    # 4. Pie chart data (entries by branch)
    cursor.execute("""
        SELECT sd.branch AS category, COUNT(*) AS num_entries
        FROM exit_entry_logs eel
        JOIN mac_roll mr ON eel.mac = mr.mac
        JOIN student_data sd ON mr.rollno = sd.rollno
        WHERE eel.entry_datetime >= CURDATE() - INTERVAL 30 DAY
        GROUP BY sd.branch;
    """)
    pie_chart_data = cursor.fetchall()

    # 5. Hour-wise entries (past 30 days)
    cursor.execute("""
        SELECT HOUR(entry_datetime) AS hr, COUNT(*) AS num_entries
        FROM exit_entry_logs
        WHERE entry_datetime >= CURDATE() - INTERVAL 30 DAY
        GROUP BY hr
        ORDER BY hr;
    """)
    hourwise_entries = cursor.fetchall()

    # 6. Heatmap: weekday vs. hour
    cursor.execute("""
        SELECT DAYOFWEEK(entry_datetime) AS weekday, HOUR(entry_datetime) AS hr, COUNT(*) AS count
        FROM exit_entry_logs
        WHERE entry_datetime >= CURDATE() - INTERVAL 30 DAY
        GROUP BY weekday, hr;
    """)
    heatmap_data = cursor.fetchall()

    # 7. Last 5 Entries
    cursor.execute("""
        SELECT mr.rollno, eel.entry_datetime, eel.exit_datetime,
               ROUND(TIMESTAMPDIFF(MINUTE, eel.exit_datetime, eel.entry_datetime)) AS duration
        FROM exit_entry_logs eel
        JOIN mac_roll mr ON eel.mac = mr.mac
        ORDER BY eel.exit_datetime DESC
        LIMIT 5;
    """)
    last_entries = cursor.fetchall()
    
    #8 OUTSIDE STUDENTS COUNT
    cursor.execute("SELECT COUNT(*) AS count FROM exit_entry_logs WHERE entry_datetime IS NULL")
    outside_count = cursor.fetchone()["count"]
    # print("lol", outside_count)

    conn.close()

    return jsonify({
        'entries7days': entries_7days,
        'exits7days': exits_7days,
        'prevdayEntries': prevday_entries,
        'prevdayExits': prevday_exits,
        'avgTimeByCategory': avg_time_by_category,
        'entryCountByCategory': entry_count_by_category,
        'hourwiseCategoryEntries': hourwise_category_entries,
        'pieChartData': pie_chart_data,
        'hourwiseEntries': hourwise_entries,
        'heatmapData': heatmap_data,
        'lastEntries': last_entries,
        "outsideStudentsCount": outside_count
    })

from flask import Flask, request, render_template

# from flask import request  # Add this at the top if not already imported

@app.route('/form')
def form_page():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT rollno, mac FROM mac_roll ORDER BY rollno")
    entries = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('form.html', entries=entries)

@app.route('/submit', methods=['POST'])
def handle_submission():
    rollno = request.form['rollno'].strip()
    raw_mac = request.form['mac'].strip()
    mac = raw_mac  # you can modify normalize_mac logic here if needed

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT * FROM mac_roll WHERE rollno = %s", (rollno,))
        existing_roll = cursor.fetchone()

        if existing_roll:
            if existing_roll['mac'] == mac:
                msg = "MAC already exists with roll number."
            else:
                cursor.execute("UPDATE mac_roll SET mac = %s WHERE rollno = %s", (mac, rollno))
                conn.commit()
                msg = "MAC updated for roll number."
        else:
            cursor.execute("SELECT rollno FROM mac_roll WHERE mac = %s", (mac,))
            existing_mac = cursor.fetchone()

            if existing_mac:
                msg = f"MAC already exists with roll number: {existing_mac['rollno']}"
                return render_template('result.html', message=msg, rollno=existing_mac['rollno'], mac=mac)

            cursor.execute("INSERT INTO mac_roll (rollno, mac) VALUES (%s, %s)", (rollno, mac))
            conn.commit()
            msg = "New record added successfully."

        cursor.execute("SELECT * FROM mac_roll WHERE rollno = %s", (rollno,))
        final_record = cursor.fetchone()
        return render_template('result.html', message=msg, rollno=final_record['rollno'], mac=final_record['mac'])

    except mysql.connector.Error as err:
        conn.rollback()
        if err.errno == 1062:
            msg = "Duplicate entry error - either roll number or MAC already exists"
        else:
            msg = "Database error: " + str(err)
        return render_template('result.html', message=msg, rollno=rollno, mac=mac)
    
    finally:
        cursor.close()
        conn.close()
    



    
@app.route('/student_form', methods=['GET'])
def student_form():
    return render_template('student_form.html')

@app.route('/student_submit', methods=['POST'])
def student_submit():
    rollno = request.form['rollno'].strip()
    year = request.form['year'].strip()
    branch = request.form['branch'].strip()
    stream = request.form['stream'].strip()
    hostel = request.form['hostel'].strip()
    gender = request.form['gender'].strip()

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT * FROM student_data WHERE rollno = %s", (rollno,))
        existing = cursor.fetchone()

        if existing:
            msg = "Student already exists."
        else:
            cursor.execute("""
                INSERT INTO student_data (rollno, year, branch, stream, hostel, gender)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (rollno, year, branch, stream, hostel, gender))
            conn.commit()
            msg = "Student registered successfully."

        return render_template('student_result.html', message=msg, rollno=rollno, year=year, branch=branch, stream=stream, hostel=hostel, gender=gender)

    except Exception as e:
        conn.rollback()
        return render_template('student_result.html', message="Error: " + str(e), rollno=rollno, year=year, branch=branch, stream=stream, hostel=hostel, gender=gender)
    finally:
        cursor.close()
        conn.close()



if __name__ == '__main__':
    app.run(debug=True)
