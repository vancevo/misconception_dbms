-- Bảng lưu trữ dataset chính (UnifiedRecord)
CREATE TABLE IF NOT EXISTS unified_records (
    sample_id VARCHAR(50) PRIMARY KEY,
    source_dataset VARCHAR(50),
    original_id VARCHAR(100),
    question_id VARCHAR(50),
    
    domain VARCHAR(50),
    subdomain VARCHAR(50),
    difficulty VARCHAR(20),
    
    question TEXT,
    reference_answer TEXT,
    student_answer TEXT,
    alternative_reference_answers JSONB,
    
    score_raw FLOAT,
    score_normalized FLOAT,
    label_2way VARCHAR(20),
    label_3way VARCHAR(20),
    label_5way VARCHAR(50),
    
    misconception_tags JSONB,
    misconception_inventory JSONB,
    missing_concepts JSONB,
    extra_incorrect_claims JSONB,
    key_concepts JSONB,
    
    feedback_short TEXT,
    feedback_detailed TEXT,
    
    split VARCHAR(20),
    is_synthetic BOOLEAN,
    is_human_annotated BOOLEAN,
    annotation_confidence FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bảng lưu trữ cấu hình và kết quả của các lần chạy Clustering
CREATE TABLE IF NOT EXISTS clustering_results (
    experiment_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    embedding_strategy VARCHAR(50),
    clustering_method VARCHAR(50),
    granularity VARCHAR(50),
    
    silhouette_score FLOAT,
    nmi_score FLOAT,
    ari_score FLOAT,
    purity_score FLOAT,
    
    n_clusters INTEGER,
    parameters JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
