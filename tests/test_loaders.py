"""Unit tests for per-source raw loaders.

Tests each loader with representative fixture files and verifies:
- Each loader produces UnifiedRecord instances
- sample_id prefixes are correct (SEB_, MOH_, GEN_, SCR_)
- Key fields are populated correctly
- Data_Scraping records have all usability flags set to false
- Malformed rows are skipped with warnings logged
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.data.loaders import (
    load_data_generate,
    load_data_scraping,
    load_mohler,
    load_scientsbank,
)
from src.data.schema import UnifiedRecord


# ── SciEntsBank fixtures & tests ──────────────────────────────────────

SCIENTSBANK_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<questions>
  <question id="Q_PHOTO" module="biology">
    <questionText>What is photosynthesis?</questionText>
    <referenceAnswer>Plants convert light energy into chemical energy.</referenceAnswer>
    <studentAnswer id="SA_1" accuracy="correct">Plants use light to make food.</studentAnswer>
    <studentAnswer id="SA_2" accuracy="incorrect">Plants eat soil.</studentAnswer>
  </question>
</questions>
"""

SCIENTSBANK_MALFORMED_XML = "<broken><xml"


def _make_scientsbank_dir(tmp_path: Path) -> Path:
    """Create a minimal SciEntsBank directory with one split."""
    seb_dir = tmp_path / "scientsbank"
    train_dir = seb_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "q1.xml").write_text(SCIENTSBANK_XML, encoding="utf-8")
    return seb_dir


class TestLoadScientsbank:
    def test_produces_unified_records(self, tmp_path: Path) -> None:
        seb_dir = _make_scientsbank_dir(tmp_path)
        records = load_scientsbank(seb_dir)
        assert len(records) == 2
        assert all(isinstance(r, UnifiedRecord) for r in records)

    def test_sample_id_prefix(self, tmp_path: Path) -> None:
        seb_dir = _make_scientsbank_dir(tmp_path)
        records = load_scientsbank(seb_dir)
        assert all(r.sample_id.startswith("SEB_") for r in records)

    def test_fields_populated(self, tmp_path: Path) -> None:
        seb_dir = _make_scientsbank_dir(tmp_path)
        records = load_scientsbank(seb_dir)
        r = records[0]
        assert r.source_dataset == "scientsbank"
        assert r.question_id == "Q_PHOTO"
        assert r.question == "What is photosynthesis?"
        assert r.reference_answer == "Plants convert light energy into chemical energy."
        assert r.student_answer == "Plants use light to make food."
        assert r.label_5way == "correct"
        assert r.split == "train"
        assert r.is_human_annotated is True

    def test_second_record_label(self, tmp_path: Path) -> None:
        seb_dir = _make_scientsbank_dir(tmp_path)
        records = load_scientsbank(seb_dir)
        assert records[1].label_5way == "incorrect"
        assert records[1].student_answer == "Plants eat soil."

    def test_malformed_xml_skipped(self, tmp_path: Path, caplog) -> None:
        seb_dir = tmp_path / "scientsbank"
        split_dir = seb_dir / "train"
        split_dir.mkdir(parents=True)
        (split_dir / "bad.xml").write_text(SCIENTSBANK_MALFORMED_XML, encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            records = load_scientsbank(seb_dir)
        assert len(records) == 0
        assert any("Malformed XML" in m for m in caplog.messages)

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        records = load_scientsbank(tmp_path / "nonexistent")
        assert records == []


# ── MohlerASAG fixtures & tests ──────────────────────────────────────

MOHLER_CSV_CONTENT = """\
question_id,question,reference_answer,student_answer,score,other
Q1,What is an array?,A contiguous block of memory.,A block of memory.,4.0,3.5
Q1,What is an array?,A contiguous block of memory.,I don't know.,1.0,0.5
Q2,Define a pointer.,A variable holding a memory address.,It points to stuff.,2.5,3.0
"""


def _make_mohler_dir(tmp_path: Path) -> Path:
    mohler_dir = tmp_path / "mohler"
    mohler_dir.mkdir(parents=True)
    (mohler_dir / "data.csv").write_text(MOHLER_CSV_CONTENT, encoding="utf-8")
    return mohler_dir


class TestLoadMohler:
    def test_produces_unified_records(self, tmp_path: Path) -> None:
        mohler_dir = _make_mohler_dir(tmp_path)
        records = load_mohler(mohler_dir)
        assert len(records) == 3
        assert all(isinstance(r, UnifiedRecord) for r in records)

    def test_sample_id_prefix(self, tmp_path: Path) -> None:
        mohler_dir = _make_mohler_dir(tmp_path)
        records = load_mohler(mohler_dir)
        assert all(r.sample_id.startswith("MOH_") for r in records)

    def test_score_raw_averaged(self, tmp_path: Path) -> None:
        mohler_dir = _make_mohler_dir(tmp_path)
        records = load_mohler(mohler_dir)
        # First row: score=4.0, other=3.5 → avg 3.75
        assert records[0].score_raw == pytest.approx(3.75)
        # Second row: score=1.0, other=0.5 → avg 0.75
        assert records[1].score_raw == pytest.approx(0.75)

    def test_fields_populated(self, tmp_path: Path) -> None:
        mohler_dir = _make_mohler_dir(tmp_path)
        records = load_mohler(mohler_dir)
        r = records[0]
        assert r.source_dataset == "mohler"
        assert r.question_id == "Q1"
        assert r.question == "What is an array?"
        assert r.reference_answer == "A contiguous block of memory."
        assert r.student_answer == "A block of memory."
        assert r.domain == "computer_science"
        assert r.is_human_annotated is True

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        records = load_mohler(tmp_path / "nonexistent")
        assert records == []


# ── Data_Generate tests (using actual CSV) ────────────────────────────

DATA_GENERATE_CSV = Path("data-generate.csv")


@pytest.mark.skipif(
    not DATA_GENERATE_CSV.exists(),
    reason="data-generate.csv not found at workspace root",
)
class TestLoadDataGenerate:
    def test_produces_unified_records(self) -> None:
        records = load_data_generate(DATA_GENERATE_CSV)
        assert len(records) > 0
        assert all(isinstance(r, UnifiedRecord) for r in records)

    def test_sample_id_prefix(self) -> None:
        records = load_data_generate(DATA_GENERATE_CSV)
        assert all(r.sample_id.startswith("GEN_") for r in records)

    def test_source_dataset(self) -> None:
        records = load_data_generate(DATA_GENERATE_CSV)
        assert all(r.source_dataset == "data_generate" for r in records)

    def test_key_fields_populated(self) -> None:
        records = load_data_generate(DATA_GENERATE_CSV)
        r = records[0]
        assert r.question != ""
        assert r.reference_answer != ""
        assert r.student_answer != ""
        assert r.domain != ""
        assert r.question_id != ""

    def test_labels_present(self) -> None:
        records = load_data_generate(DATA_GENERATE_CSV)
        r = records[0]
        assert r.label_5way is not None
        assert r.label_3way is not None
        assert r.label_2way is not None

    def test_adversarial_fields_preserved(self) -> None:
        records = load_data_generate(DATA_GENERATE_CSV)
        adv_records = [r for r in records if r.perturbation_type is not None]
        assert len(adv_records) > 0
        for r in adv_records:
            assert r.is_adversarial is True

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        records = load_data_generate(tmp_path / "nonexistent.csv")
        assert records == []


class TestLoadDataGenerateFixture:
    """Tests using a small inline CSV fixture for Data_Generate."""

    FIXTURE_CSV = (
        "instance_id,question_id,domain,subdomain,difficulty,split,question,"
        "reference_answer,alternative_reference_answers,key_concepts,"
        "misconception_inventory,student_answer,student_answer_style,"
        "lexical_overlap_level,semantic_correctness_score_0_5,label_5way,"
        "label_3way,label_2way,misconception_tags,"
        "misconception_span_rationale,missing_concepts,"
        "extra_incorrect_claims,feedback_short,feedback_detailed,"
        "feedback_type,feedback_tone,adversarial_variant_of,"
        "perturbation_type,robustness_notes,annotation_confidence\n"
        'FIX_001,Q01,biology,cells,easy,train,What is a cell?,'
        '"The basic unit of life.","[]","[""cell"",""life""]","[]",'
        '"A cell is the smallest unit of life.",concise,high,5,correct,'
        'correct,correct,"[]",Good.,"[]","[]",Great!,Well done.,praise,'
        "tutor_like,,,none,0.95\n"
    )

    def test_fixture_produces_record(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "gen.csv"
        csv_path.write_text(self.FIXTURE_CSV, encoding="utf-8")
        records = load_data_generate(csv_path)
        assert len(records) == 1
        r = records[0]
        assert r.sample_id.startswith("GEN_")
        assert r.source_dataset == "data_generate"
        assert r.question == "What is a cell?"
        assert r.score_raw == pytest.approx(5.0)
        assert r.label_5way == "correct"
        assert r.annotation_confidence == pytest.approx(0.95)

    def test_malformed_row_skipped(self, tmp_path: Path, caplog) -> None:
        # Create a CSV with a row that has an invalid difficulty value
        bad_csv = (
            "instance_id,question_id,domain,subdomain,difficulty,split,"
            "question,reference_answer,alternative_reference_answers,"
            "key_concepts,misconception_inventory,student_answer,"
            "student_answer_style,lexical_overlap_level,"
            "semantic_correctness_score_0_5,label_5way,label_3way,"
            "label_2way,misconception_tags,misconception_span_rationale,"
            "missing_concepts,extra_incorrect_claims,feedback_short,"
            "feedback_detailed,feedback_type,feedback_tone,"
            "adversarial_variant_of,perturbation_type,robustness_notes,"
            "annotation_confidence\n"
            'BAD_001,Q01,bio,cells,INVALID_DIFFICULTY,train,Q?,Ref.,"[]",'
            '"[]","[]",Ans.,concise,low,3,correct,correct,correct,"[]",'
            'ok.,"[]","[]",ok,ok,praise,tutor_like,,,none,0.9\n'
        )
        csv_path = tmp_path / "bad_gen.csv"
        csv_path.write_text(bad_csv, encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            records = load_data_generate(csv_path)
        # The row should be skipped because "INVALID_DIFFICULTY" is not valid
        assert len(records) == 0
        assert any("Malformed" in m or "skipping" in m for m in caplog.messages)


# ── Data_Scraping tests (using actual JSON) ───────────────────────────

DATA_SCRAPING_JSON = Path("data-scraping.json")


@pytest.mark.skipif(
    not DATA_SCRAPING_JSON.exists(),
    reason="data-scraping.json not found at workspace root",
)
class TestLoadDataScraping:
    def test_produces_unified_records(self) -> None:
        records = load_data_scraping(DATA_SCRAPING_JSON)
        assert len(records) > 0
        assert all(isinstance(r, UnifiedRecord) for r in records)

    def test_sample_id_prefix(self) -> None:
        records = load_data_scraping(DATA_SCRAPING_JSON)
        assert all(r.sample_id.startswith("SCR_") for r in records)

    def test_source_dataset(self) -> None:
        records = load_data_scraping(DATA_SCRAPING_JSON)
        assert all(r.source_dataset == "data_scraping" for r in records)

    def test_usability_flags_all_false(self) -> None:
        records = load_data_scraping(DATA_SCRAPING_JSON)
        for r in records:
            assert r.usable_for_grading is False
            assert r.usable_for_feedback is False
            assert r.usable_for_misconception_mining is False
            assert r.usable_for_robustness_eval is False

    def test_is_synthetic_false(self) -> None:
        records = load_data_scraping(DATA_SCRAPING_JSON)
        assert all(r.is_synthetic is False for r in records)

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        records = load_data_scraping(tmp_path / "nonexistent.json")
        assert records == []


class TestLoadDataScrapingFixture:
    """Tests using a small inline JSON fixture for Data_Scraping."""

    FIXTURE_DATA = [
        {
            "id": "physics_1_1",
            "questions": "What is velocity?",
            "reference_answer": "Rate of change of displacement.",
            "student_answer": "",
            "label": "openstax_college-physics",
        },
        {
            "id": "physics_1_2",
            "questions": "Define acceleration.",
            "reference_answer": "Rate of change of velocity.",
            "student_answer": "",
            "label": "openstax_college-physics",
        },
    ]

    def test_fixture_produces_records(self, tmp_path: Path) -> None:
        json_path = tmp_path / "scraping.json"
        json_path.write_text(json.dumps(self.FIXTURE_DATA), encoding="utf-8")
        records = load_data_scraping(json_path)
        assert len(records) == 2

    def test_fixture_fields(self, tmp_path: Path) -> None:
        json_path = tmp_path / "scraping.json"
        json_path.write_text(json.dumps(self.FIXTURE_DATA), encoding="utf-8")
        records = load_data_scraping(json_path)
        r = records[0]
        assert r.sample_id == "SCR_00001"
        assert r.source_dataset == "data_scraping"
        assert r.original_id == "physics_1_1"
        assert r.question == "What is velocity?"
        assert r.reference_answer == "Rate of change of displacement."
        assert r.usable_for_grading is False
        assert r.usable_for_feedback is False
        assert r.usable_for_misconception_mining is False
        assert r.usable_for_robustness_eval is False

    def test_malformed_json_returns_empty(self, tmp_path: Path, caplog) -> None:
        json_path = tmp_path / "bad.json"
        json_path.write_text("{not valid json", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            records = load_data_scraping(json_path)
        assert records == []
        assert any("Error reading" in m for m in caplog.messages)

    def test_non_list_json_returns_empty(self, tmp_path: Path, caplog) -> None:
        json_path = tmp_path / "obj.json"
        json_path.write_text('{"key": "value"}', encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            records = load_data_scraping(json_path)
        assert records == []
        assert any("not a list" in m for m in caplog.messages)
