"""지식그래프 시드 + 온톨로지 스키마 (Tier 0 + 함정 5종 범위).

L1 스코프: Person·Source·Dataset(Tier 0) + 함정 발화에 필요한 Paper/Method/Metric/CitedWork.
의도적으로 제외(L2 범위): Concept 레이어, addresses/applies_to 엣지.

노드는 attr `type`, 엣지는 attr `rel`(관계 타입)을 가진다. 방향이 중요하다(검증기가 검사).
"""
import networkx as nx

# ── 온톨로지 스키마 (검증기·plan 프롬프트의 단일 출처) ─────────────────────
# Concept = L2 Tier-1. 지금은 M4(flagship 멀티홉) 1건만 성립시키는 최소 집합:
# det_vs_learned_verification(P3 발견) + 메서드 kind 2개(중립 typing). 다른 토픽 개념은 보류.
NODE_TYPES = ["Paper", "Person", "Source", "Dataset", "Method", "Metric", "CitedWork", "Concept"]

# relation 이름 -> (src_type, dst_type). 방향이 곧 스키마(단일방향).
RELATIONS = {
    "authored_by":  ("Paper", "Person"),
    "uses":         ("Paper", "Dataset"),
    "derived_from": ("Dataset", "Source"),
    "employs":      ("Paper", "Method"),
    "evaluated_by": ("Paper", "Metric"),
    "cites":        ("Paper", "CitedWork"),   # P3만. 논문 간 cite는 없음
    # ── L2 Tier-1 (M4용). 전부 단일방향. 결론('전이됨')은 박지 않는 중립 사실들 ──
    "addresses":    ("Paper", "Concept"),     # 논문이 그 개념을 다룸 (불일치는 Paper→Concept로 통일)
    "is_a":         ("Method", "Concept"),    # 메서드의 종류(선택적 typing)
    "contrasts":    ("Concept", "Concept"),   # 개념이 두 kind를 대조(P3 §4.6 critic ablation의 중립 사실)
}

# 노드 정의: id -> (type, 표시용 속성)
NODES = {
    # Paper
    "P1": ("Paper", {"title": "청년 고립 탐지 (HA-XGB)"}),
    "P2": ("Paper", {"title": "산업 전력 이상탐지 (업종 조건화)"}),
    "P3": ("Paper", {"title": "SQL 생성 + EXPLAIN 자가수정 (4B SLM)"}),
    # Person
    "손병훈": ("Person", {}),
    "이상금": ("Person", {}),
    "정민성": ("Person", {}),
    # Source / Dataset
    "KEPCO": ("Source", {"full": "KEPCO Green Button (8업종, 2020-2022)"}),
    "KEPCO-AD":  ("Dataset", {"records": "57.06M", "shape": "시계열"}),
    "KEPCO-SQL": ("Dataset", {"records": "3.04M", "shape": "단일테이블"}),
    # Method
    "LSTM-AE": ("Method", {}),
    "XGBoost": ("Method", {}),
    "Shannon-entropy": ("Method", {}),
    "NBEATSx": ("Method", {}),
    "LoRA": ("Method", {}),
    "EXPLAIN-self-correction": ("Method", {}),
    # Metric
    "F1": ("Metric", {}),
    "VUS-PR": ("Metric", {}),
    "FCR": ("Metric", {"value": "90/77% compliance"}),
    # CitedWork (P3만)
    "CHASE-SQL": ("CitedWork", {}),
    "Self-Refine": ("CitedWork", {}),
    "Reflexion": ("CitedWork", {}),
    # Concept (L2 Tier-1, M4 최소집합)
    "deterministic_vs_learned_verification": ("Concept", {"note": "P3 §4.6: critic 80% 통과 vs EXPLAIN 100%"}),
    "learned_discriminator": ("Concept", {}),
    "deterministic_verifier": ("Concept", {}),
}

# 엣지: (src, rel, dst)
EDGES = [
    ("P1", "authored_by", "손병훈"), ("P1", "authored_by", "이상금"),
    ("P2", "authored_by", "손병훈"), ("P2", "authored_by", "이상금"), ("P2", "authored_by", "정민성"),
    ("P3", "authored_by", "손병훈"), ("P3", "authored_by", "이상금"), ("P3", "authored_by", "정민성"),

    ("P2", "uses", "KEPCO-AD"), ("P3", "uses", "KEPCO-SQL"),
    ("KEPCO-AD", "derived_from", "KEPCO"), ("KEPCO-SQL", "derived_from", "KEPCO"),

    ("P1", "employs", "LSTM-AE"), ("P1", "employs", "XGBoost"), ("P1", "employs", "Shannon-entropy"),
    ("P2", "employs", "NBEATSx"),
    ("P3", "employs", "LoRA"), ("P3", "employs", "EXPLAIN-self-correction"),

    ("P1", "evaluated_by", "F1"), ("P2", "evaluated_by", "VUS-PR"), ("P3", "evaluated_by", "FCR"),

    ("P3", "cites", "CHASE-SQL"), ("P3", "cites", "Self-Refine"), ("P3", "cites", "Reflexion"),

    # ── L2 Tier-1 (M4) ──
    ("P3", "addresses", "deterministic_vs_learned_verification"),
    ("XGBoost", "is_a", "learned_discriminator"),
    ("LSTM-AE", "is_a", "learned_discriminator"),
    ("NBEATSx", "is_a", "learned_discriminator"),
    ("EXPLAIN-self-correction", "is_a", "deterministic_verifier"),
    # contrasts: 딱 두 엣지에서 멈춤(다른 개념으로 확장 금지)
    ("deterministic_vs_learned_verification", "contrasts", "learned_discriminator"),
    ("deterministic_vs_learned_verification", "contrasts", "deterministic_verifier"),
    # (LoRA·Shannon-entropy 등은 판별기/검증기 아님 → is_a 무태깅, 선택적임을 보임)
]


def build_kg():
    """온톨로지대로 NetworkX DiGraph 구성."""
    g = nx.MultiDiGraph()  # 같은 쌍에 여러 관계가 올 수 있어 Multi
    for nid, (ntype, attrs) in NODES.items():
        g.add_node(nid, type=ntype, **attrs)
    for src, rel, dst in EDGES:
        g.add_edge(src, dst, rel=rel)
    return g


def schema_text():
    """plan 노드 LLM 프롬프트에 넣을 사람이 읽는 스키마."""
    lines = ["NODE TYPES: " + ", ".join(NODE_TYPES), "", "RELATIONS (방향 src -> dst):"]
    for rel, (s, d) in RELATIONS.items():
        note = "  (P3만; 논문 간 cite 없음)" if rel == "cites" else ""
        lines.append(f"  {rel}: {s} -> {d}{note}")
    lines.append("")
    lines.append("ENTITIES (타입별):")
    by_type = {}
    for nid, (ntype, _) in NODES.items():
        by_type.setdefault(ntype, []).append(nid)
    for ntype in NODE_TYPES:
        lines.append(f"  {ntype}: " + ", ".join(by_type.get(ntype, [])))
    return "\n".join(lines)
