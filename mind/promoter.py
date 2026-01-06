"""
Path: mind/promoter.py
Promoter gate: core-mind pull-request logic (stub critic + uncertainty + PII strip)
"""
import re, os, uuid
from heart.uncertainty_gate import UncertaintyGate
from mind.vector_store import AetherVectorStore
from loguru import logger

# --- config ---
PROMOTE_SURPRISE   = float(os.getenv("PROMOTE_SURPRISE", 0.55))
PROMOTE_FLOURISH   = float(os.getenv("PROMOTE_FLOURISH", 0.35))
UNCERTAINTY_CUT    = 0.35

# --- stub critic (rules only) ---
BS_PATTERNS = {
    "physics": re.compile(r"\bperpetual motion|free energy|flat earth\b", re.I),
    "health":  re.compile(r"\bdrink bleach|essential oils cure cancer\b", re.I),
    "code":    re.compile(r"\brm -rf /|fork bomb\b", re.I),
}

def _strip_pii(text: str) -> str:
    # naive scrubber – emails / phones / IPs
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "<PHONE>", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", text)
    return text

class Promoter:
    def __init__(self, store: AetherVectorStore, gate: UncertaintyGate):
        self.store = store
        self.gate  = gate

    async def nugget_maybe_promote(self, user_id: str, raw_text: str, surprise: float, flourish: float) -> str:
        if surprise < PROMOTE_SURPRISE or flourish < PROMOTE_FLOURISH:
            return "below_threshold"
        cleaned = _strip_pii(raw_text)
        if len(cleaned) < 30:
            return "too_short"

        # 1. stub critic
        for cat, pat in BS_PATTERNS.items():
            if pat.search(cleaned):
                logger.info(f"Promoter rejected {cat} bs")
                return "critic_fail"

        # 2. uncertainty vs core_universal
        ctx_vec, _ = self.store.query_context(cleaned, namespace="core_universal", top_k=1)
        state_vec  = [0.0]*1024 if not ctx_vec else ctx_vec[0]
        should_block, uncertainty = self.gate.should_block(state_vec)
        if should_block or uncertainty > UNCERTAINTY_CUT:
            logger.info(f"Promoter rejected – uncertainty {uncertainty:.2f}")
            return "uncertainty_fail"

        # 3. promote
        metadata = {"source_user": user_id, "promoted_at": logger.now().isoformat()}
        self.store.upsert_knowledge(cleaned, namespace="core_universal", metadata=metadata)
        logger.success(f"Promoter accepted insight from {user_id}")
        return "promoted"

    def record_plan_outcome(self, plan_id: str, success: bool):
        text = f"Plan {plan_id} outcome: {'success' if success else 'failed'}"
        metadata = {"type": "plan", "id": plan_id, "success": success}
        self.store.upsert_knowledge(text, namespace="core_universal", metadata=metadata)
