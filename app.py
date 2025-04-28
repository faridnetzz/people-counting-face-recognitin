import os
import sqlite3
import csv
from flask import jsonify, Flask, render_template, send_file, request, redirect, url_for, Response
from io import StringIO, BytesIO
import pandas as pd

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True


def get_filtered_data(gender=None, date=None):
    conn = sqlite3.connect('face_logs.db')
    cursor = conn.cursor()
    query = "SELECT * FROM detections"
    filters = []
    params = []

    if gender:
        filters.append("gender = ?")
        params.append(gender)
    if date:
        filters.append("DATE(timestamp) = ?")
        params.append(date)

    if filters:
        query += " WHERE " + " AND ".join(filters)

    query += " ORDER BY timestamp DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows


@app.route('/stats')
def stats():
    conn = sqlite3.connect('face_logs.db')
    c = conn.cursor()
    c.execute("SELECT strftime('%H', timestamp) as hour, COUNT(*) FROM detections GROUP BY hour")
    data = c.fetchall()
    conn.close()
    hours = [row[0] for row in data]
    counts = [row[1] for row in data]
    return render_template('stats.html', hours=hours, counts=counts)


@app.route('/')
def index():
    image_folder = os.path.join(app.static_folder, 'face_snapshots')
    os.makedirs(image_folder, exist_ok=True)
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: os.path.getmtime(os.path.join(image_folder, x)),
        reverse=True
    )
    recent_images = image_files[:10]

    gender = request.args.get('gender')
    date = request.args.get('date')
    rows = get_filtered_data(gender, date)

    total = len(rows)
    male = sum(1 for r in rows if r[2] == 'Male')
    female = sum(1 for r in rows if r[2] == 'Female')

    return render_template('index.html', data=rows, total=total, male=male, female=female, recent_images=recent_images)


@app.route('/export/csv')
def export_csv():
    gender = request.args.get('gender')
    date = request.args.get('date')
    rows = get_filtered_data(gender, date)

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Timestamp', 'Gender', 'X', 'Y', 'Width', 'Height', 'Image Path'])
    cw.writerows(rows)

    return Response(
        si.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=detections.csv'}
    )


@app.route('/export/excel')
def export_excel():
    gender = request.args.get('gender')
    date = request.args.get('date')
    rows = get_filtered_data(gender, date)

    df = pd.DataFrame(rows, columns=['ID', 'Timestamp', 'Gender', 'X', 'Y', 'Width', 'Height', 'Image Path'])
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Detections')

    output.seek(0)
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     download_name='detections.xlsx', as_attachment=True)


@app.route('/download/<filename>')
def download_image(filename):
    filepath = os.path.join('static', 'face_snapshots', filename)
    return send_file(filepath, as_attachment=True)


@app.route('/api/data')
def api_data():
    conn = sqlite3.connect('face_logs.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, gender, image_path, x || ',' || y || ',' || width || ',' || height FROM detections ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    data = []
    for row in rows:
        img_url = url_for('static', filename=row[2]) if row[2] else None
        data.append({
            "time": row[0],
            "gender": row[1],
            "image": img_url,
            "coordinates": row[3]
        })
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
