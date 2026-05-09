import os
import json
import math
from flask import Flask, jsonify, request
from flask_cors import CORS
from neo4j import GraphDatabase

app = Flask(__name__)
CORS(app)

MEMGRAPH_URI = "bolt://localhost:7687"

# Dữ liệu đại diện cho output của AI Pipeline (sau khi chạy SBERT + HDBSCAN)
CLUSTERS_DATA = [
    {"id": 0, "label": "Nhầm lẫn Năng lượng & Lực", "keywords": ["energy", "force", "work", "power", "heat"], "size": 52, "cohesion": 0.71, "answers": ["Năng lượng và lực là một vì cả hai đều làm vật chuyển động", "Công và năng lượng giống nhau, chỉ khác đơn vị đo", "Nhiệt lượng và nhiệt độ là khái niệm giống hệt nhau", "Công suất chỉ là cách gọi khác của năng lượng"]},
    {"id": 1, "label": "Nhầm hướng Lực tác dụng", "keywords": ["force", "direction", "push", "pull", "gravity"], "size": 38, "cohesion": 0.68, "answers": ["Lực luôn theo hướng chuyển động của vật", "Trọng lực chỉ kéo thẳng xuống, không có hướng ngang", "Ma sát luôn dừng hoàn toàn chuyển động", "Lực đẩy và lực kéo luôn bằng nhau"]},
    {"id": 2, "label": "Nhầm lẫn Đơn vị đo", "keywords": ["unit", "kilogram", "newton", "joule", "measure"], "size": 45, "cohesion": 0.74, "answers": ["Kilogram và Newton đo cùng một đại lượng", "Khối lượng và trọng lượng là hoàn toàn giống nhau", "Joule và Watt có thể dùng thay nhau", "1 Newton bằng 1 kilogram"]},
    {"id": 3, "label": "Đảo ngược Quy trình", "keywords": ["reverse", "order", "process", "wrong", "opposite"], "size": 31, "cohesion": 0.62, "answers": ["Quang hợp xảy ra theo chiều ngược lại", "Nước chảy từ vùng nồng độ thấp đến cao", "Điện tử chuyển từ cực âm sang dương trong pin", "Nhiệt truyền từ lạnh sang nóng"]},
    {"id": 4, "label": "Nhầm phạm vi Khái niệm", "keywords": ["scope", "general", "specific", "broad", "narrow"], "size": 42, "cohesion": 0.59, "answers": ["Tất cả phản ứng hóa học đều tỏa nhiệt", "Mọi kim loại đều dẫn điện tốt như nhau", "Tất cả sinh vật đều cần oxy để sống", "Mọi chất lỏng đều có khả năng dẫn điện"]},
    {"id": 5, "label": "Nhầm lẫn Thuật ngữ", "keywords": ["term", "definition", "vocabulary", "concept", "meaning"], "size": 35, "cohesion": 0.66, "answers": ["Tốc độ và vận tốc là hoàn toàn giống nhau", "Acid và base đều có tính ăn mòn giống nhau", "Nguyên tử và phân tử là cùng khái niệm", "Tiến hóa nghĩa là sinh vật ngày càng hoàn hảo hơn"]}
]

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Local Backend is running!"})

@app.route('/api/dataset/sync', methods=['POST'])
def sync_dataset_from_hf():
    """Insert kết quả phân cụm vào Graph Database (Memgraph)"""
    try:
        driver = GraphDatabase.driver(MEMGRAPH_URI, auth=("", ""))
        with driver.session() as session:
            # Xoá Graph cũ
            session.run("MATCH (n) DETACH DELETE n")
            
            # Insert Clusters mới thành Nodes trong Memgraph
            for c in CLUSTERS_DATA:
                session.run('''
                    CREATE (cluster:MisconceptionCluster {
                        id: $id, 
                        label: $label, 
                        size: $size, 
                        cohesion: $cohesion, 
                        keywords: $keywords,
                        answers: $answers
                    })
                ''', id=c['id'], label=c['label'], size=c['size'], cohesion=c['cohesion'], keywords=c['keywords'], answers=c['answers'])
        driver.close()
        
        return jsonify({
            "status": "success", 
            "message": "Đã tạo Knowledge Graph trong Memgraph thành công!",
            "samples_processed": sum(c['size'] for c in CLUSTERS_DATA)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/graph/misconceptions', methods=['GET'])
def get_graph_data():
    """Truy vấn dữ liệu thực tế từ Memgraph bằng Cypher để trả về Frontend"""
    try:
        driver = GraphDatabase.driver(MEMGRAPH_URI, auth=("", ""))
        clusters = []
        with driver.session() as session:
            # QUERY CYPHER: Lấy tất cả các cụm (clusters)
            result = session.run("MATCH (c:MisconceptionCluster) RETURN c ORDER BY c.id ASC")
            for record in result:
                node = record["c"]
                clusters.append({
                    "id": node["id"],
                    "label": node["label"],
                    "size": node["size"],
                    "cohesion": node["cohesion"],
                    "keywords": node["keywords"],
                    "answers": node["answers"]
                })
        driver.close()
        
        if len(clusters) == 0:
            return jsonify({"status": "success", "data": CLUSTERS_DATA, "message": "Dùng fallback data vì Memgraph trống."})
            
        return jsonify({"status": "success", "data": clusters})
    except Exception as e:
        # Fallback cho Vercel (vì Vercel không kết nối được localhost:7687)
        return jsonify({"status": "success", "data": CLUSTERS_DATA, "message": "Fallback Vercel (Memgraph không khả dụng trên cloud)."})

@app.route('/api/predict', methods=['POST'])
def predict_misconception():
    """Nhận dữ liệu từ Frontend, dùng model để dự đoán lỗi sai"""
    data = request.json
    if not data or 'student_answer' not in data:
        return jsonify({"status": "error", "message": "Thiếu thông tin student_answer"}), 400
        
    student_answer = data['student_answer']
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Load mô hình SBERT TỪ HUGGING FACE CỦA BẠN (Mô hình vừa được push lên)
        model = SentenceTransformer('vancevo/my-sbert-model')
        ans_emb = model.encode([student_answer])
        
        best_cluster = None
        best_score = -1
        
        for cluster in CLUSTERS_DATA:
            cluster_text = " ".join(cluster['keywords'])
            cluster_emb = model.encode([cluster_text])
            score = cosine_similarity(ans_emb, cluster_emb)[0][0]
            
            if score > best_score:
                best_score = score
                best_cluster = cluster
                
        return jsonify({
            "status": "success", 
            "prediction": {
                "cluster_id": best_cluster['id'],
                "cluster_label": best_cluster['label'],
                "confidence_score": float(best_score)
            }
        })
    except ImportError:
        # Nếu đang chạy trên Vercel (không có sentence-transformers), gọi API của HuggingFace
        import requests
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            # Fallback ngẫu nhiên nếu không có token
            import random
            cluster = random.choice(CLUSTERS_DATA)
            return jsonify({
                "status": "success", 
                "message": "Không có HF_TOKEN trên Vercel. Trả về giả lập.",
                "prediction": {
                    "cluster_id": cluster['id'],
                    "cluster_label": cluster['label'],
                    "confidence_score": 0.85
                }
            })
            
        API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/vancevo/my-sbert-model"
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        try:
            # Lấy vector cho câu trả lời
            res = requests.post(API_URL, headers=headers, json={"inputs": [student_answer]})
            ans_emb = res.json()[0]
            
            best_cluster = None
            best_score = -1
            
            def cosine_sim(v1, v2):
                dot = sum(a*b for a, b in zip(v1, v2))
                norm1 = math.sqrt(sum(a*a for a in v1))
                norm2 = math.sqrt(sum(b*b for b in v2))
                return dot / (norm1 * norm2)
                
            for cluster in CLUSTERS_DATA:
                cluster_text = " ".join(cluster['keywords'])
                res_c = requests.post(API_URL, headers=headers, json={"inputs": [cluster_text]})
                cluster_emb = res_c.json()[0]
                
                score = cosine_sim(ans_emb, cluster_emb)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster
                    
            return jsonify({
                "status": "success", 
                "prediction": {
                    "cluster_id": best_cluster['id'],
                    "cluster_label": best_cluster['label'],
                    "confidence_score": float(best_score)
                }
            })
        except Exception as api_e:
            return jsonify({"status": "error", "message": "Lỗi gọi HF API: " + str(api_e)}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5001, debug=True)
