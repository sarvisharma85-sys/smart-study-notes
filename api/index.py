from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"

@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'Please enter text to summarize.'}), 400
    
    orig_words = len(text.split())
    if orig_words < 20:
        return jsonify({'error': 'Text is too short. Please enter at least 20 words.'}), 400

    max_len = max(25, int(orig_words * 0.5))
    min_len = max(10, int(orig_words * 0.2))

    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_len,
            "min_length": min_len,
            "do_sample": False
        }
    }

    try:
        response = requests.post(API_URL, json=payload)
        res_data = response.json()
        
        if isinstance(res_data, dict) and 'error' in res_data:
            return jsonify({'error': f"Model loading, please try again in a few seconds: {res_data['error']}"}), 503

        summary_text = res_data[0]['summary_text']
        summary_words = len(summary_text.split())
        reduction_pct = round(((orig_words - summary_words) / orig_words) * 100, 2)

        return jsonify({
            'summary': summary_text,
            'orig_count': orig_words,
            'summary_count': summary_words,
            'reduction_pct': reduction_pct
        })
    except Exception as e:
        return jsonify({'error': f"Summarization failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()
