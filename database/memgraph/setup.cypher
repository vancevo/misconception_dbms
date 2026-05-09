// 1. Tạo các Index để tăng tốc độ truy vấn
CREATE INDEX ON :Question(id);
CREATE INDEX ON :StudentAnswer(id);
CREATE INDEX ON :MisconceptionCluster(cluster_id);
CREATE INDEX ON :Concept(name);

// 2. Mẫu Insert Data (Demo thủ công, thực tế sẽ insert bằng Python Backend)

// Tạo câu hỏi
MERGE (q:Question {id: 'Q015'})
ON CREATE SET q.domain = 'physics', q.difficulty = 'medium', q.text = 'What is the relationship between force and acceleration?';

// Tạo Khái niệm
MERGE (c1:Concept {name: 'F = ma relationship'});
MERGE (c2:Concept {name: 'role of mass'});

// Liên kết Câu hỏi kiểm tra Khái niệm
MATCH (q:Question {id: 'Q015'}), (c1:Concept {name: 'F = ma relationship'})
MERGE (q)-[:TESTS]->(c1);

// Tạo cụm Lỗi sai (Cluster)
MERGE (cluster:MisconceptionCluster {cluster_id: 1})
ON CREATE SET cluster.keywords = ['force', 'acceleration', 'same', 'unit'];

// Tạo Sinh viên trả lời
MERGE (s:StudentAnswer {id: 'GEN_00042'})
ON CREATE SET s.text = 'Force and acceleration are the same thing, just measured in different units.', s.score = 0.1;

// Liên kết sinh viên trả lời câu hỏi
MATCH (s:StudentAnswer {id: 'GEN_00042'}), (q:Question {id: 'Q015'})
MERGE (s)-[:ANSWERS]->(q);

// Sinh viên bị thiếu khái niệm
MATCH (s:StudentAnswer {id: 'GEN_00042'}), (c1:Concept {name: 'F = ma relationship'})
MERGE (s)-[:MISSING_CONCEPT]->(c1);

// Câu trả lời thuộc về nhóm lỗi sai số 1
MATCH (s:StudentAnswer {id: 'GEN_00042'}), (cluster:MisconceptionCluster {cluster_id: 1})
MERGE (s)-[:BELONGS_TO]->(cluster);
