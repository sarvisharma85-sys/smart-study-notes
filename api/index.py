from flask import Flask, request, jsonify
import re

app = Flask(__name__)

def generate_extractive_summary(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    if len(sentences) <= 2:
        return text

    words = re.findall(r'\w+', text.lower())
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 
        'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
        'up', 'down', 'of', 'off', 'over', 'under', 'again', 'further', 'then', 
        'once', 'this', 'that', 'these', 'those', 'it', 'its', 'however', 'also'
    }
    
    freq = {}
    for word in words:
        if word not in stop_words:
            freq[word] = freq.get(word, 0) + 1

    scored_sentences = []
    for sentence in sentences:
        score = sum(freq.get(w.lower(), 0) for w in re.findall(r'\w+', sentence))
        scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    num_sentences = max(1, len(sentences) // 2)
    top_sentences = [s[1] for s in scored_sentences[:num_sentences]]

    selected_in_order = [s for s in sentences if s in top_sentences]
    return " ".join(selected_in_order)

@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Please enter text to summarize.'}), 400

        orig_words = len(text.split())
        if orig_words < 20:
            return jsonify({'error': 'Text is too short. Please enter at least 20 words.'}), 400

        summary_text = generate_extractive_summary(text)
        summary_words = len(summary_text.split())
        reduction_pct = round(((orig_words - summary_words) / orig_words) * 100, 2)

        return jsonify({
            'summary': summary_text,
            'orig_count': orig_words,
            'summary_count': summary_words,
            'reduction_pct': reduction_pct
        })
    except Exception as e:
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()
