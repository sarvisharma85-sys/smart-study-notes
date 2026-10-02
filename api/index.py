from flask import Flask, request, jsonify
import requests
import re

app = Flask(__name__)

# Updated Hugging Face router URL
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

def fallback_summarize(text):
    """Smart local summarizer fallback if Hugging Face is down/loading."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= 2:
        return text
    
    # Calculate word frequency
    words = re.findall(r'\w+', text.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'of', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'this', 'that', 'these', 'those', 'it', 'its'}
    freq = {}
    for word in words:
        if word not in stop_words:
            freq[word] = freq.get(word, 0) + 1

    # Score sentences
    scored_sentences = []
    for sentence in sentences:
        score = sum(freq.get(w.lower(), 0) for w in re.findall(r'\w+', sentence))
        scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    num_sentences = max(1, len(sentences) // 2)
    selected = [s[1] for s in scored_sentences[:num_sentences]]
    
    # Preserve original sentence order
    selected_in_order = [s for s in sentences if s in selected]
    return " ".join(selected_in_order)

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
        "parameters": {"max_length": max_len, "min_length": min_len, "do_sample": False},
        "options": {"wait_for_model": True}
    }

    summary_text = None

    # Try Hugging Face Inference API
    try:
        response = requests.post(HF_API_URL, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if response.status_code == 200:
            res_data = response.json()
            if isinstance(res_data, list) and len(res_data) > 0 and 'summary_text' in res_data[0]:
                summary_text = res_data[0]['summary_text']
    except Exception:
        pass

    # Fallback locally if API call failed or timed out
    if not summary_text:
        summary_text = fallback_summarize(text)

    summary_words = len(summary_text.split())
    reduction_pct = round(((orig_words - summary_words) / orig_words) * 100, 2)

    return jsonify({
        'summary': summary_text,
        'orig_count': orig_words,
        'summary_count': summary_words,
        'reduction_pct': reduction_pct
    })

if __name__ == '__main__':
    app.run()
