from sentence_transformers import SentenceTransformer, util
import re


model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_keywords(text, top_k=20):
    
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq = {}

    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [w for w, c in sorted_words[:top_k]]

    return keywords



def calculate_similarity(resume_text, jd_text):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(resume_emb, jd_emb).item()
    return similarity



def find_missing_skills(resume_keywords, jd_keywords):
    return [kw for kw in jd_keywords if kw not in resume_keywords]



def ats_score(similarity, missing_count):
    """
    Final score = 70% similarity + 30% skill coverage
    """
    similarity_score = similarity * 70
    skill_score = max(0, 30 - missing_count)  # lose 1 point per missing skill

    return int(min(100, similarity_score + skill_score))
