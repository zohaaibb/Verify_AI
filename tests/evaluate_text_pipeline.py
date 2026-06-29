"""
Evaluate the full text analysis pipeline (orchestrator) against 100 labeled claims.

Usage:
    1. Start the Flask backend: cd backend && python app.py
    2. Run this script:   python tests/evaluate_text_pipeline.py

The script registers a test user, logs in, submits each claim via POST /submit,
collects the final verdict, and computes:
    - Accuracy (on decided / non-SUSPICIOUS verdicts)
    - Precision / Recall / F1 on FAKE class
    - SUSPICIOUS rate
    - Confusion matrix
"""

import json
import sys
import time
import requests
from dataclasses import dataclass, field
from typing import Optional

BASE = "http://127.0.0.1:5000"
API = f"{BASE}/api"


# ── Ground-truth dataset ──────────────────────────────────────────────────────
# Label: 1 = FAKE, 0 = REAL
# These are real-world-style claims for testing. Replace with your own curated set.
GROUND_TRUTH = [
    # ── FAKE claims (label = 1) ──────────────────────────────────────────────
    ("Shocking: Doctors hate this one simple trick to cure all diseases", 1),
    ("Government hiding aliens in Area 51 – leaked photos confirmed", 1),
    ("New World Order plans to microchip all citizens by 2025", 1),
    ("Drink bleach to cure COVID-19, says anonymous doctor", 1),
    ("Pakistan will host 2030 FIFA World Cup, confirms FIFA source", 1),
    ("Imran Khan announces new political party 'Tehreek-e-Insaaf 2.0'", 1),
    ("100% guaranteed weight loss pill – lose 20kg in one week", 1),
    ("Bill Gates admits vaccines contain tracking microchips", 1),
    ("Earth to be destroyed by asteroid next month, NASA confirms", 1),
    ("Pakistan's economy grows 500% in one month – IMF shocked", 1),
    ("Secret WhatsApp message reveals PM's resignation plan", 1),
    ("Miracle water cures cancer – big pharma doesn't want you to know", 1),
    ("Hollywood star dies in car crash – family confirms (not true)", 1),
    ("Supreme Court secretly bans all opposition parties in Pakistan", 1),
    ("Scientists discover that oxygen causes aging – new study", 1),
    ("Pakistan wins 5 Cricket World Cups in a row – ICC investigation", 1),
    ("Facebook to charge $10/month starting next month – must share to opt out", 1),
    ("Saudi Arabia announces Mecca will be moved to a new location", 1),
    ("India and Pakistan merge into one country – UN resolution passed", 1),
    ("10 million jobs lost in one day – government hiding statistics", 1),
    ("Election results completely fabricated – leaked evidence shows", 1),
    ("This one fruit destroys cancer cells in hours – doctors shocked", 1),
    ("World War III starts tomorrow – leaked Pentagon memo", 1),
    ("Antarctica melting will flood all coastal cities by next year", 1),
    ("Aliens land in Karachi – video goes viral (CGI but claimed real)", 1),
    ("COVID-19 vaccines cause infertility in women – study claims", 1),
    ("Mobile towers cause coronavirus – 5G conspiracy resurfaces", 1),
    ("Drink urine every morning – Ayurvedic doctor says it cures everything", 1),
    ("Snake venom cures baldness – new research from China", 1),
    ("Pakistan's GDP surpasses United States – World Bank report", 1),
    ("All banks will close permanently by December – internal memo leaked", 1),
    ("Earthquake will destroy Lahore on Friday – psychic predicts", 1),
    ("You won't believe what this actor said about the government", 1),
    ("Pfizer CEO admits vaccine was never tested on humans", 1),
    ("Make money from home earning Rs. 500,000 per day – join now", 1),
    ("President to declare martial law tomorrow – sources say", 1),
    ("Pakistan developing invisibility cloak – defence official confirms", 1),
    ("Ancient temple discovered under Parliament House in Islamabad", 1),
    ("Elon Musk buys Pakistan's entire national debt – Twitter post", 1),
    ("Sun will turn blue for three days – NASA warning", 1),
    ("Kashmir to become independent within 30 days – UN sources", 1),
    ("Borderless world announced – no passports needed from 2025", 1),
    ("Free solar panels for every household – government scheme (scam)", 1),
    ("One million Pakistanis to be deported from UAE tomorrow", 1),
    ("CIA confirms involvement in Pakistan's internal politics – leaked file", 1),
    ("Click this link to claim your free iPhone 15 – limited offer", 1),
    ("Sindh to become separate country – referendum announced", 1),
    ("Pakistan announces nuclear test – world on edge (false alarm)", 1),
    ("Mars colony established – Elon Musk's secret project revealed", 1),
    ("All petrol stations to close for one week – fuel crisis warning (hoax)", 1),

    # ── REAL claims (label = 0) ──────────────────────────────────────────────
    ("Prime Minister meets with IMF delegation to discuss loan program", 0),
    ("Pakistan announces new education policy for primary schools", 0),
    ("Pakistan cricket team arrives in England for test series", 0),
    ("Supreme Court hears petition on election delays", 0),
    ("Federal budget allocates more funds for healthcare sector", 0),
    ("World Bank approves $500 million loan for infrastructure projects", 0),
    ("UNICEF launches vaccination drive in rural areas", 0),
    ("Stock market closes higher on positive economic indicators", 0),
    ("Heavy rainfall expected in Karachi over the next two days", 0),
    ("Government launches digital payment system for public services", 0),
    ("Pakistan and China sign new trade agreement", 0),
    ("Oil prices drop amid global economic slowdown concerns", 0),
    ("NASA successfully launches new space telescope", 0),
    ("WHO announces new guidelines for pandemic preparedness", 0),
    ("European Union imposes new sanctions on Russia", 0),
    ("Federal Reserve keeps interest rates unchanged", 0),
    ("Japan allocates aid for flood-affected regions in Pakistan", 0),
    ("New variant of COVID-19 detected in South Africa", 0),
    ("Pakistan reports trade deficit decrease of 15% this quarter", 0),
    ("Education minister announces scholarship program for girls", 0),
    ("Parliament passes bill on data protection and privacy", 0),
    ("Pakistan Railways introduces new freight service", 0),
    ("SBP keeps policy rate unchanged at record low", 0),
    ("KP government launches health insurance for low-income families", 0),
    ("Pakistan's remittances increase by 10% in first quarter", 0),
    ("ADB approves $300 million for energy sector reforms", 0),
    ("Pakistan's exports reach highest level in five years", 0),
    ("Climate change conference concludes with new emissions targets", 0),
    ("Gold prices fall by Rs. 2,000 per tola in domestic market", 0),
    ("Pakistan Super League season to start in February", 0),
    ("New cancer treatment shows promising results in clinical trials", 0),
    ("Facebook announces new privacy features for users", 0),
    ("Pakistan's IT exports cross $2 billion mark", 0),
    ("Supreme Court establishes new environmental protection bench", 0),
    ("UN peacekeeping mission in Africa completes 10 years", 0),
    ("World Food Programme provides aid to Afghan refugees", 0),
    ("Pakistan's textile industry creates 50,000 new jobs", 0),
    ("Electric vehicle charging stations to be installed across major cities", 0),
    ("Pakistan and Saudi Arabia sign investment protection agreement", 0),
    ("New bridge inaugurated in Gwadar to boost trade", 0),
    ("Rainfall deficit reported in Sindh – water authorities concerned", 0),
    ("National database registration authority launches mobile app", 0),
    ("Pakistan's foreign exchange reserves rise to $15 billion", 0),
    ("Supreme Court bars dual nationals from holding public office", 0),
    ("Punjab government starts tree plantation drive", 0),
    ("Pakistan achieves 90% polio vaccination coverage", 0),
    ("New anti-terrorism legislation introduced in parliament", 0),
    ("SECP introduces new reforms for ease of doing business", 0),
    ("Pakistan's inflation rate drops to single digit", 0),
    ("CPEC enters second phase with new industrial cooperation", 0),
]

# ── Orchestrator verdicts are: SAFE / LOW THREAT / MEDIUM THREAT / HIGH THREAT
# We map them:  SAFE & LOW THREAT → REAL (0), MEDIUM & HIGH THREAT → FAKE (1)
VERDICT_MAP = {
    "SAFE":         0,
    "LOW THREAT":   0,
    "MEDIUM THREAT": 1,
    "HIGH THREAT":  1,
}


@dataclass
class Result:
    claim: str
    truth: int          # 0=REAL, 1=FAKE
    verdict: Optional[str] = None
    score: Optional[int] = None
    modules_ok: int = 0
    modules_total: int = 0
    error: Optional[str] = None


def register_user(session, username: str) -> Optional[str]:
    """Register a test user and return the JWT token."""
    resp = session.post(f"{API}/auth/register", json={
        "username": username,
        "email": f"{username}@test.com",
        "password": "testpass123",
        "subscription": "enterprise",
    })
    if resp.status_code == 201:
        data = resp.json()
        return data.get("access_token")
    # Already registered — try logging in
    resp = session.post(f"{API}/auth/login", json={
        "username": username,
        "password": "testpass123",
    })
    if resp.status_code == 200:
        data = resp.json()
        return data.get("access_token")
    print(f"  Auth failed: {resp.status_code} {resp.text}")
    return None


def evaluate_claim(session, token: str, claim: str, modules: list) -> dict:
    """Submit a claim and return the orchestrator result."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = session.post(
        f"{API}/analysis/submit",
        json={"input_type": "text", "text": claim, "modules": modules},
        headers=headers,
        timeout=120,
    )
    if resp.status_code == 200:
        data = resp.json()
        results = data.get("results", {})
        summary = results.get("summary", {})
        return {
            "verdict": summary.get("verdict"),
            "score": summary.get("overall_threat_score"),
            "modules_ok": summary.get("modules_succeeded", 0),
            "modules_total": summary.get("modules_completed", 0),
            "raw": results,
        }
    return {"error": f"HTTP {resp.status_code}: {resp.text}"}


def print_confusion_matrix(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    print(f"\n{'='*50}")
    print(f"{'':>20} {'PREDICTED':>20}")
    print(f"{'':>20} {'FAKE':>10} {'REAL':>10}")
    print(f"{'ACTUAL FAKE':>20} {tp:>10} {fn:>10}")
    print(f"{'ACTUAL REAL':>20} {fp:>10} {tn:>10}")
    print(f"{'='*50}")
    print(f"  Total: {total}")
    if total > 0:
        accuracy = (tp + tn) / total * 100
        print(f"  Accuracy: {accuracy:.2f}%")
    if tp + fp > 0:
        precision = tp / (tp + fp) * 100
        print(f"  Precision (FAKE): {precision:.2f}%")
    else:
        precision = 0.0
    if tp + fn > 0:
        recall = tp / (tp + fn) * 100
        print(f"  Recall (FAKE): {recall:.2f}%")
    else:
        recall = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"  F1 (FAKE): {f1:.2f}%")
    print(f"{'='*50}\n")
    return accuracy, precision, recall, f1


def main():
    username = f"eval_user_{int(time.time())}"

    # Which modules to run for text
    MODULES = ["ai", "osint", "trusted", "scraper", "wikipedia"]

    session = requests.Session()
    session.headers.update({"User-Agent": "TruthLens-Eval/1.0"})

    # 1. Authenticate
    print("Authenticating...")
    token = register_user(session, username)
    if not token:
        print("FATAL: Could not authenticate. Is the backend running?")
        print(f"  Start it with: cd E:\\Verify_AI\\backend && python app.py")
        sys.exit(1)
    print("  Token obtained.")

    # 2. Verify connectivity
    headers = {"Authorization": f"Bearer {token}"}
    resp = session.get(f"{API}/analysis/modules", headers=headers)
    if resp.status_code != 200:
        print(f"FATAL: /modules returned {resp.status_code}. Backend running?")
        sys.exit(1)

    # 3. Run evaluation
    results: list[Result] = []
    total = len(GROUND_TRUTH)
    print(f"\nEvaluating {total} claims with modules: {MODULES}\n")

    for i, (claim, truth) in enumerate(GROUND_TRUTH, 1):
        res = Result(claim=claim, truth=truth)
        try:
            out = evaluate_claim(session, token, claim, MODULES)
            if "error" in out:
                res.error = out["error"]
            else:
                res.verdict = out["verdict"]
                res.score = out["score"]
                res.modules_ok = out["modules_ok"]
                res.modules_total = out["modules_total"]
        except Exception as e:
            res.error = str(e)

        results.append(res)

        # Progress
        status = res.verdict or "ERROR"
        s = f"[{i:>3}/{total}] {status:>15} | {claim[:60]}"
        if res.error:
            s += f"  ❌ {res.error}"
        print(s)

        # Small delay to avoid overwhelming the server
        time.sleep(0.2)

    # 4. Compute metrics
    decided = [r for r in results if r.verdict is not None]
    errors = [r for r in results if r.verdict is None]

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total claims:         {len(results)}")
    print(f"  Successful verdicts:  {len(decided)}")
    print(f"  Errors/Timeouts:      {len(errors)}")

    if errors:
        print(f"\n  Errors breakdown:")
        for r in errors:
            print(f"    ❌ {r.claim[:60]} -> {r.error}")

    # Confusion matrix on decided claims
    tp = sum(1 for r in decided if r.truth == 1 and VERDICT_MAP.get(r.verdict) == 1)
    fp = sum(1 for r in decided if r.truth == 0 and VERDICT_MAP.get(r.verdict) == 1)
    fn = sum(1 for r in decided if r.truth == 1 and VERDICT_MAP.get(r.verdict) == 0)
    tn = sum(1 for r in decided if r.truth == 0 and VERDICT_MAP.get(r.verdict) == 0)

    print_confusion_matrix(tp, fp, fn, tn)

    # 5. Summary JSON
    summary = {
        "total": len(results),
        "decided": len(decided),
        "errors": len(errors),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy_pct": round((tp + tn) / len(decided) * 100, 2) if decided else 0,
        "precision_pct": round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0,
        "recall_pct": round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0,
        "f1_pct": round(2 * (tp/(tp+fp)) * (tp/(tp+fn)) / ((tp/(tp+fp)) + (tp/(tp+fn))) * 100, 2)
        if (tp + fp) > 0 and (tp + fn) > 0 else 0,
        "modules_used": MODULES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to eval_results.json")

    # Individual results
    detailed = []
    for r in results:
        detailed.append({
            "claim": r.claim,
            "truth": "FAKE" if r.truth else "REAL",
            "verdict": r.verdict,
            "score": r.score,
            "error": r.error,
        })
    with open("eval_results_detailed.json", "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"  Detailed results saved to eval_results_detailed.json")


if __name__ == "__main__":
    main()
