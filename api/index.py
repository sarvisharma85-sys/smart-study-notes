from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Primary & Fallback public endpoints
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

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

    try:
        response = requests.post(API_URL, json=payload, timeout=25)
        
        # Check if response returned valid JSON
        try:
            res_data = response.json()
        except Exception:
            return jsonify({'error': 'Hugging Face API returned a non-JSON response. Please try again in a few seconds.'}), 502

        if isinstance(res_data, dict) and 'error' in res_data:
            return jsonify({'error': f"Model loading: {res_data['error']}. Try again in a few seconds."}), 503

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
        
        return jsonify({'error': 'Unexpected response format from AI model.'}), 500

    except Exception as e:
        return jsonify({'error': f"Summarization request failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()
