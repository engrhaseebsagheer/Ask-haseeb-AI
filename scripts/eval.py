"""Run the test questions through the real pipeline and report which pass.

Run from the project root after any change to prompts, models, chunking or documents:
    python scripts/eval.py

Each question lists groups of words; an answer passes when it contains at least one word from
every group. A question marked expect_refusal passes when the assistant says it does not know.
Costs about a cent per run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv()
from backend.app.services.rag_service import rag_answer  # noqa: E402

cases = json.loads((Path(__file__).parent / "eval_questions.json").read_text(encoding="utf-8"))
passed = 0
for case in cases:
    answer = rag_answer(case["question"], "short")["answer"]
    low = answer.lower()
    if case.get("expect_refusal"):
        ok = "don't have enough information" in low
    else:
        ok = all(any(word.lower() in low for word in group) for group in case["expect_any"])
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'}  {case['question']}")
    if not ok:
        print(f"      answer: {answer[:300]}")

print(f"\n{passed} of {len(cases)} passed")
sys.exit(0 if passed == len(cases) else 1)
