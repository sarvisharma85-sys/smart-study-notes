from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Primary and fallback endpoints
ENDPOINTS = [
    "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
    "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn",
    "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
]

@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.get_json() or {}
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
        },
        "options": {
            "wait_for_model": True
        }
    }

    last_error = ""
    for url in ENDPOINTS:
        try:
            response = requests.post(
                url, 
                json=payload, 
                headers={"User-Agent": "Mozilla/5.0"}, 
                timeout=15
            )
            
            # Check if response is valid JSON
            try:
                res_data = response.json()
            except Exception:
                continue

            if isinstance(res_data, dict) and 'error' in res_data:
                last_error = res_data['error']
                continue

            if isinstance(res_data, list) and len(res_data) > 0 and 'summary_text' in res_data[0]:
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
            last_error = str(e)
            continue

    return jsonify({'error': f"AI model busy or loading. Please try again in 5 seconds. Details: {last_error}"}), 503

if __name__ == '__main__':
    app.run()
