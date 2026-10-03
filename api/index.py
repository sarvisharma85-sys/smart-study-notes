from flask import Flask, request, jsonify
import re

app = Flask(__name__)

def generate_extractive_summary(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
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

        # Split into individual paragraphs (separated by blank lines or line breaks)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]
        
        paragraph_observations = []
        full_summary_list = []

        for p in paragraphs:
            p_words = len(p.split())
            if p_words == 0:
                continue
            
            p_summary = generate_extractive_summary(p)
            p_sum_words = len(p_summary.split())
            p_reduction = round(((p_words - p_sum_words) / p_words) * 100, 2) if p_words > 0 else 0

            paragraph_observations.append({
                'orig_words': p_words,
                'summary_words': p_sum_words,
                'reduction_pct': p_reduction,
                'summary': p_summary
            })
            full_summary_list.append(p_summary)

        full_summary = " ".join(full_summary_list)
        total_summary_words = len(full_summary.split())
        total_reduction = round(((orig_words - total_summary_words) / orig_words) * 100, 2)

        return jsonify({
            'summary': full_summary,
            'orig_count': orig_words,
            'summary_count': total_summary_words,
            'reduction_pct': total_reduction,
            'paragraph_observations': paragraph_observations
        })
    except Exception as e:
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()
