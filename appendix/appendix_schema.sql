CREATE TABLE datasets (
    dataset_id BIGSERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    source_type VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE learning_resources (
    resource_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT REFERENCES datasets(dataset_id),
    title TEXT,
    domain VARCHAR(100),
    chapter VARCHAR(255),
    source_url TEXT
);

CREATE TABLE questions (
    question_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT REFERENCES datasets(dataset_id),
    resource_id BIGINT REFERENCES learning_resources(resource_id),
    question_text TEXT NOT NULL,
    reference_answer TEXT,
    difficulty VARCHAR(50)
);

CREATE TABLE student_answers (
    answer_id BIGSERIAL PRIMARY KEY,
    question_id BIGINT REFERENCES questions(question_id),
    sample_id VARCHAR(100) UNIQUE NOT NULL,
    student_answer TEXT NOT NULL,
    split VARCHAR(50),
    is_synthetic BOOLEAN DEFAULT FALSE
);

CREATE TABLE grading_labels (
    label_id BIGSERIAL PRIMARY KEY,
    answer_id BIGINT REFERENCES student_answers(answer_id),
    score_raw NUMERIC(8,4),
    score_normalized NUMERIC(8,4),
    label_3way VARCHAR(50),
    label_5way VARCHAR(100)
);

CREATE TABLE misconception_inventory (
    misconception_id BIGSERIAL PRIMARY KEY,
    tag VARCHAR(150) UNIQUE NOT NULL,
    name VARCHAR(255),
    description TEXT,
    domain VARCHAR(100),
    expected_concept TEXT
);

CREATE TABLE misconception_annotations (
    annotation_id BIGSERIAL PRIMARY KEY,
    answer_id BIGINT REFERENCES student_answers(answer_id),
    misconception_id BIGINT REFERENCES misconception_inventory(misconception_id),
    missing_concepts JSONB,
    extra_incorrect_claims JSONB,
    annotation_confidence NUMERIC(5,4)
);

CREATE TABLE experiment_runs (
    run_id VARCHAR(100) PRIMARY KEY,
    strategy VARCHAR(50),
    method VARCHAR(50),
    granularity VARCHAR(50),
    model_name VARCHAR(255),
    random_seed INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE embedding_records (
    embedding_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES experiment_runs(run_id),
    answer_id BIGINT REFERENCES student_answers(answer_id),
    embedding VECTOR(384),
    umap_5d JSONB,
    umap_x NUMERIC(12,6),
    umap_y NUMERIC(12,6)
);

CREATE TABLE clusters (
    cluster_pk BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES experiment_runs(run_id),
    cluster_id INT,
    cluster_label TEXT,
    size INT,
    is_noise BOOLEAN DEFAULT FALSE
);

CREATE TABLE cluster_memberships (
    membership_id BIGSERIAL PRIMARY KEY,
    cluster_pk BIGINT REFERENCES clusters(cluster_pk),
    answer_id BIGINT REFERENCES student_answers(answer_id),
    representative_rank INT
);

CREATE TABLE cluster_keywords (
    keyword_id BIGSERIAL PRIMARY KEY,
    cluster_pk BIGINT REFERENCES clusters(cluster_pk),
    keyword TEXT,
    weight NUMERIC(12,6)
);

CREATE TABLE evaluation_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES experiment_runs(run_id),
    metric_name VARCHAR(100),
    metric_value NUMERIC(12,6)
);

CREATE TABLE demo_exports (
    export_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES experiment_runs(run_id),
    export_path TEXT,
    export_format VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);