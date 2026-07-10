import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

csv_path = "/Users/chihjuihsu/Documents/python_script/TXF_analysis/xau/data/XAU_1m_data.csv"
labels_path = "/Users/chihjuihsu/Documents/python_script/TXF_analysis/xau/data/labels.json"

# Cache unique dates on startup to ensure rapid UI responsiveness
print("Scanning CSV to pre-load available dates...")
cached_dates = []
try:
    if os.path.exists(csv_path):
        unique_set = set()
        with open(csv_path, 'r', encoding='utf-8') as f:
            f.readline() # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                if parts and parts[0]:
                    date_part = parts[0].split(' ')[0]
                    if len(date_part.split('.')) == 3:
                        unique_set.add(date_part)
        cached_dates = sorted(list(unique_set))
        print(f"Pre-loaded {len(cached_dates)} available trading dates.")
    else:
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
except Exception as e:
    print(f"Warning: Failed to pre-load dates due to {e}", file=sys.stderr)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dates', methods=['GET'])
def get_dates():
    return jsonify(cached_dates)

@app.route('/api/data', methods=['GET'])
def get_data():
    target_date = request.args.get('date')
    if not target_date:
        return jsonify({"error": "Missing 'date' parameter"}), 400
        
    normalized_target = target_date.replace('-', '.').replace('/', '.')
    
    # Load target date data with 50-row preceding buffer for MA5 rolling calculation
    buffer = []
    target_rows = []
    header = None
    buffer_size = 50
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            header = header_line.split(';')
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                if len(parts) < 6:
                    continue
                    
                dt_str = parts[0]
                date_part = dt_str.split(' ')[0]
                
                if date_part == normalized_target:
                    target_rows.append(parts)
                elif len(target_rows) > 0:
                    break
                else:
                    buffer.append(parts)
                    if len(buffer) > buffer_size:
                        buffer.pop(0)
                        
        if not target_rows:
            return jsonify({"error": f"Date '{target_date}' not found in data"}), 404
            
        all_rows = buffer + target_rows
        df = pd.DataFrame(all_rows, columns=header)
        
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        
        # Calculate Low-based MA5
        df['ma5'] = df['Low'].rolling(window=5).mean()
        
        # Slice off the buffer
        target_df = df.iloc[len(buffer):].copy()
        target_df = target_df.replace({np.nan: None})
        
        data_list = target_df.to_dict(orient='records')
        return jsonify(data_list)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/labels', methods=['GET'])
def get_labels_for_date():
    target_date = request.args.get('date')
    if not target_date:
        return jsonify([])
    normalized_target = target_date.replace('-', '.').replace('/', '.')
    
    if not os.path.exists(labels_path):
        return jsonify([])
        
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            labels = data.get("labels", [])
            date_labels = [l for l in labels if l.get("date") == normalized_target]
            return jsonify(date_labels)
    except Exception:
        return jsonify([])

@app.route('/api/save', methods=['POST'])
def save_label():
    try:
        label_data = request.json
        if not label_data:
            return jsonify({"error": "Missing JSON payload"}), 400
            
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {"labels": []}
        else:
            data = {"labels": []}
            
        if "labels" not in data:
            data["labels"] = []
            
        # Upsert logic (insert new or update matching date & end_time)
        target_date = label_data.get("date")
        target_end_time = label_data.get("end_time")
        
        existing_idx = -1
        for idx, l in enumerate(data["labels"]):
            if l.get("date") == target_date and l.get("end_time") == target_end_time:
                existing_idx = idx
                break
                
        if existing_idx != -1:
            data["labels"][existing_idx] = label_data
            print(f"Updated label '{label_data.get('label')}' at {target_end_time} for {target_date}.")
        else:
            data["labels"].append(label_data)
            print(f"Saved new label '{label_data.get('label')}' at {target_end_time} for {target_date}.")
            
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        return jsonify({"status": "success", "total_labels": len(data["labels"])})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete', methods=['POST'])
def delete_label():
    try:
        req = request.json
        target_date = req.get("date")
        target_end_time = req.get("end_time")
        
        if not os.path.exists(labels_path):
            return jsonify({"error": "No labels file found"}), 404
            
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        labels = data.get("labels", [])
        original_len = len(labels)
        new_labels = [l for l in labels if not (l.get("date") == target_date and l.get("end_time") == target_end_time)]
        
        if len(new_labels) == original_len:
            return jsonify({"error": "Label not found"}), 404
            
        data["labels"] = new_labels
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Deleted label at {target_end_time} for {target_date}.")
        return jsonify({"status": "success", "total_labels": len(new_labels)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
