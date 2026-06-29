# -*- coding: utf-8 -*-
"""
Evaluate Verify_AI against 200 labeled news claims (100 REAL + 100 FAKE).

Usage:
    1. Start backend: cd backend && python app.py
    2. Run:          python tests/evaluate_200_news.py

Makes API calls to POST /api/analysis/submit, collects verdicts,
and prints confusion matrix + accuracy / precision / recall / F1.
"""

import json
import sys
import time
import requests
from dataclasses import dataclass
from typing import Optional

BASE = "http://127.0.0.1:5000"
API = BASE + "/api"

# Label: 1 = FAKE, 0 = REAL
GROUND_TRUTH = [
    # ===== REAL NEWS (label=0) — sourced from verified outlets =====
    ("Israeli strikes kill 8 in Lebanon as fighting with Hezbollah intensifies despite US-led efforts to reduce tensions", 0),
    ("Russia launched a large-scale attack on Kyiv killing at least 13 and injuring over 100 across Ukraine", 0),
    ("Iran says US peace deal hinges on release of $24 billion in frozen assets, warns against renewed conflict", 0),
    ("Pakistan sets June 10 budget date as PM Shehbaz Sharif courts business elite on tax and exports", 0),
    ("2026-27 Budget: Conservative plan under IMF pressure with higher tax targets and IMF-linked constraints", 0),
    ("Finance Minister Muhammad Aurangzeb presents federal budget with total outlay of Rs 17.573 trillion for FY25-26", 0),
    ("Pakistan, Russia sign agreements to combat illegal immigration and drug trafficking", 0),
    ("NAB hands over Rs 6 billion recovered assets to Khyber Pakhtunkhwa government", 0),
    ("Sindh cabinet approves Rs 11.2 billion Karachi infrastructure package", 0),
    ("PPP and PML-N government continue budget negotiations amid unresolved concerns", 0),
    ("Summer vacation announced by Supreme Court of Pakistan for 2026", 0),
    ("Major progress made on digital payment system for passport fees in Pakistan", 0),
    ("GS Malik appointed as Director General of Police of Gujarat with immediate effect", 0),
    ("JMM nominates former minister Baidyanath Ram as candidate for Rajya Sabha seat from Jharkhand", 0),
    ("Fadnavis unveils Chhatrapati Shivaji Maharaj statue at Navi Mumbai international airport", 0),
    ("Italy marks 80 years of the Republic with military parade in Rome", 0),
    ("European Union imposes new sanctions on Russia over continued aggression in Ukraine", 0),
    ("Bulgarian government resigns after months of anti-corruption protests", 0),
    ("Myanmar airstrike on Rakhine hospital kills 30, injures dozens", 0),
    ("US Federal Reserve cuts interest rates in divided vote, signals pause and one 2026 cut as growth rebounds", 0),
    ("UN General Assembly adopts resolution on climate change obligations of states following ICJ advisory opinion", 0),
    ("Hundreds of thousands displaced in South Sudan's Jonglei State due to conflict and insecurity", 0),
    ("Trump set to nominate acting AG Todd Blanche as attorney general, sparking confirmation battle", 0),
    ("NTSB preliminary report reveals United jet that hit light pole on NJ Turnpike was just 15 feet above highway", 0),
    ("Google DeepMind launches Gemini Omni as latest multimodal AI model", 0),
    ("Google releases Gemini 3.5 - frontier intelligence with action capabilities", 0),
    ("OpenAI rolls out Lockdown Mode - optional security setting protecting against prompt injection attacks", 0),
    ("Alibaba's Qwen team releases Qwen3.7-Plus, a multimodal agent model combining visual perception, GUI operation and coding", 0),
    ("Alphabet announces $80 billion AI infrastructure investment plan", 0),
    ("Anthropic moves toward public offering, filing IPO paperwork", 0),
    ("Florida launches historic lawsuit against OpenAI", 0),
    ("Google Cloud signs compute deal with SpaceX for bridge capacity to meet surging customer demand for Gemini Enterprise", 0),
    ("SpaceX warns investors that water scarcity is now a critical risk factor for AI infrastructure", 0),
    ("Supermicro doubles down on rack-scale AI with new systems built on AMD's Helios platform", 0),
    ("Snowflake expands enterprise AI footprint with CoCo coding agents for data-centric app workflows", 0),
    ("Intel showcases chips-to-rack solutions at Computex to accelerate datacenter deliveries", 0),
    ("Cisco launches agentic platform to operate and defend critical IT infrastructure", 0),
    ("Netskope introduces AI Command Center for discovery and coordinated agentic response", 0),
    ("Akamai teams with NVIDIA to bring security inside AI factories", 0),
    ("US AI data center buildout falls behind schedule as power and permitting bottlenecks grow", 0),
    ("Europe pushes for tech sovereignty with France courting more than 110 billion euros in AI investments", 0),
    ("Paris-listed Teleperformance becomes one of Europe's most shorted stocks as hedge funds bet on AI disruption", 0),
    ("Pak-ID app now handles passport fingerprints - new verification system resolves online application issues", 0),
    ("Zong's new data centre marks key milestone in Pakistan's digital transformation", 0),
    ("WHO announces new guidelines for pandemic preparedness and response", 0),
    ("UN confirms onset of El Nino, warns of above-average temperatures globally and extreme weather", 0),
    ("World Environment Day 2026 focuses on urgent climate action with global temperatures at near-record levels", 0),
    ("GEF Assembly in Samarkand makes final push for nature finance and environmental funding", 0),
    ("UN urges all countries to bolster early warning systems as El Nino conditions develop", 0),
    ("New variant of COVID-19 detected in South Africa, WHO monitoring", 0),
    ("Scientists reveal the origin of the Euphrates river in the cradle of civilization", 0),
    ("New cancer treatment shows promising results in clinical trials at major research hospital", 0),
    ("Federal budget allocates more funds for healthcare sector with 15% increase in spending", 0),
    ("World Bank approves $500 million loan for infrastructure projects in Pakistan", 0),
    ("Pakistan stock market closes higher on positive economic indicators and investor confidence", 0),
    ("SBP keeps policy rate unchanged at record low to support economic growth", 0),
    ("Pakistan's remittances increase by 10% in first quarter of 2026", 0),
    ("Pakistan's exports reach highest level in five years with textile sector leading growth", 0),
    ("Gold prices fall by Rs 2,000 per tola in domestic market amid global economic slowdown", 0),
    ("Pakistan's IT exports cross $2 billion mark for the first time", 0),
    ("Pakistan's foreign exchange reserves rise to $15 billion", 0),
    ("Pakistan's inflation rate drops to single digit for first time in three years", 0),
    ("ADB approves $300 million for energy sector reforms in Pakistan", 0),
    ("Pakistan Railways introduces new freight service to boost trade connectivity", 0),
    ("Pakistan's textile industry creates 50,000 new jobs in first quarter", 0),
    ("SECP introduces new reforms for ease of doing business in Pakistan", 0),
    ("Report flags rising pressure on compliant taxpayers in Pakistan as FBR targets increase", 0),
    ("Electricity prices might be lower than solar power says Federal Minister Awais Leghari", 0),
    ("Oil prices drop amid global economic slowdown concerns and Iran-US talks progress", 0),
    ("Walmart adds express delivery from in-store restaurants starting June 2026", 0),
    ("Rubrik reports first quarter fiscal year 2027 financial results beating expectations", 0),
    ("FIFA praises Pakistan football revival and plans key reforms to modernize governance", 0),
    ("National Junior Championship 2026 to kick off in Peshawar with 952 athletes competing", 0),
    ("Serena Williams announces return to tennis at age 44 after four-year hiatus", 0),
    ("Liverpool close to hiring Andoni Iraola as manager to replace Arne Slot", 0),
    ("Andrew Flintoff named Sydney Thunder head coach on two-year deal", 0),
    ("Liga MX signs licensing deal with Electronic Arts to return to EA Sports FC video game", 0),
    ("NFL approves Arctos Partners' investment in Cleveland Browns, expanding team portfolio", 0),
    ("MLB The Show Mobile launches worldwide on iOS and Android", 0),
    ("PSL season 11 underway with multiple teams competing for title", 0),
    ("Parliament passes bill on data protection and privacy in Pakistan", 0),
    ("KP government launches health insurance for low-income families covering 2 million households", 0),
    ("Electric vehicle charging stations to be installed across major cities in Pakistan", 0),
    ("Pakistan and Saudi Arabia sign investment protection agreement worth $5 billion", 0),
    ("New bridge inaugurated in Gwadar to boost trade and connectivity with China", 0),
    ("Punjab government starts tree plantation drive targeting 1 billion trees", 0),
    ("CPEC enters second phase with new industrial cooperation agreements signed", 0),
    ("Pakistan and China sign new trade agreement worth $10 billion", 0),
    ("Japan allocates $50 million aid for flood-affected regions in Pakistan", 0),
    ("Supreme Court establishes new environmental protection bench in Pakistan", 0),
    ("Rainfall deficit reported in Sindh as water authorities express concern over water availability", 0),
    ("New anti-terrorism legislation introduced in Pakistan parliament", 0),
    ("Supreme Court bars dual nationals from holding public office in Pakistan", 0),
    ("Pakistan reports trade deficit decrease of 15% in current quarter", 0),
    ("Prime Minister meets with IMF delegation to discuss loan program progress", 0),
    ("Pakistan announces new education policy for primary schools focusing on STEM", 0),
    ("Pakistan cricket team arrives in England for test series against England", 0),
    ("Supreme Court hears petition on election delays in upcoming general elections", 0),
    ("National database registration authority launches mobile app for document verification", 0),
    ("Pakistan achieves 90% polio vaccination coverage nationwide", 0),

    # ===== FAKE NEWS (label=1) — fabricated but realistic =====
    ("Shocking: Doctors hate this one simple trick to cure all diseases permanently", 1),
    ("Government hiding aliens in Area 51 - leaked photos confirmed authentic by anonymous source", 1),
    ("New World Order plans to microchip all citizens by 2027 through mandatory vaccination", 1),
    ("Drink bleach to cure COVID-19, says anonymous doctor from undisclosed hospital", 1),
    ("Pakistan will host 2030 FIFA World Cup after FIFA secretly agrees to expand tournament", 1),
    ("Imran Khan announces new political party Tehreek-e-Insaaf 2.0 from prison", 1),
    ("100% guaranteed weight loss pill - lose 20kg in one week without exercise", 1),
    ("Bill Gates admits in leaked video that vaccines contain tracking microchips", 1),
    ("Earth to be destroyed by asteroid next month, NASA scientists confirm in panic", 1),
    ("Pakistan's economy grows 500% in one month - IMF economists shocked", 1),
    ("Secret WhatsApp message reveals Prime Minister's resignation plan leaked by insider", 1),
    ("Miracle water from Himalayan spring cures all forms of cancer - big pharma suppressing study", 1),
    ("Hollywood superstar dies in car crash - family confirms tragic loss", 1),
    ("Supreme Court secretly bans all opposition parties in Pakistan through midnight decree", 1),
    ("Scientists discover that oxygen causes aging - new study from controversial research lab", 1),
    ("Pakistan wins 5 Cricket World Cups in a row - ICC launches investigation into miracle run", 1),
    ("Facebook to charge $10 per month starting next month - share this to opt out", 1),
    ("Saudi Arabia announces Mecca will be relocated to new purpose-built city", 1),
    ("India and Pakistan merge into one country - UN resolution passed unanimously", 1),
    ("10 million jobs lost in one day - government hiding unemployment statistics", 1),
    ("Election results completely fabricated - leaked evidence shows widespread rigging", 1),
    ("This one fruit destroys cancer cells in hours - doctors shocked by results", 1),
    ("World War III starts tomorrow - leaked Pentagon memo reveals invasion plans", 1),
    ("Antarctica melting will flood all coastal cities by next year - UN report suppressed", 1),
    ("Aliens land in Karachi - viral video shows extraterrestrial craft but CGI experts disagree", 1),
    ("COVID-19 vaccines cause infertility in women - unpublished study claims", 1),
    ("Mobile towers cause coronavirus - 5G conspiracy resurfaces with new leaked document", 1),
    ("Drink urine every morning - Ayurvedic doctor says it cures all diseases", 1),
    ("Snake venom cures baldness - new research from secret Chinese lab", 1),
    ("Pakistan's GDP surpasses United States - World Bank report accidentally published then retracted", 1),
    ("All banks will close permanently by December - internal central bank memo leaked", 1),
    ("Earthquake will destroy Lahore on Friday - psychic prediction goes viral", 1),
    ("Pfizer CEO admits vaccine was never tested on humans before rollout", 1),
    ("President to declare martial law tomorrow - anonymous security source confirms", 1),
    ("Pakistan developing invisibility cloak - defence official confirms top secret project", 1),
    ("Ancient temple discovered under Parliament House in Islamabad - archaeologists baffled", 1),
    ("Elon Musk buys Pakistan's entire national debt - tweet confirms $200 billion purchase", 1),
    ("Sun will turn blue for three days - NASA warning ignored by mainstream media", 1),
    ("Kashmir to become independent within 30 days - UN sources confirm secret resolution", 1),
    ("Borderless world announced - no passports needed from 2027 as global government formed", 1),
    ("Free solar panels for every household - government scheme requires only processing fee", 1),
    ("One million Pakistanis to be deported from UAE tomorrow - sudden policy change", 1),
    ("CIA confirms involvement in Pakistan's internal politics - leaked classified file", 1),
    ("Click this link to claim your free iPhone 16 - limited time offer for first 1000 users", 1),
    ("Sindh to become separate country - referendum announced by separatist leaders", 1),
    ("Pakistan announces surprise nuclear test - world on edge as radiation detected", 1),
    ("Mars colony established - Elon Musk's secret SpaceX project revealed by insider", 1),
    ("All petrol stations to close for one week - fuel crisis warning is hoax but goes viral", 1),
    ("UN declares Pakistan a failed state - resolution passed in emergency session", 1),
    ("Queen Elizabeth faked her own death - palace cover-up exposed by whistleblower", 1),
    ("Secret tunnel network discovered under Islamabad connecting all government buildings", 1),
    ("WHO confirms that eating eggs causes heart disease in all humans - new study retracted", 1),
    ("Pakistan Army takes over government in bloodless coup - midnight announcement", 1),
    ("Facebook CEO Mark Zuckerberg converts to Islam after visiting Pakistan", 1),
    ("NASA discovers parallel universe where time runs backwards - classified report", 1),
    ("Diamond reserves worth $1 trillion discovered in Karachi - government stays silent", 1),
    ("All mobile networks to be shut down for 48 hours nationwide - interior ministry alert", 1),
    ("Groundbreaking study finds that chocolate can replace all medicines - funded by candy industry", 1),
    ("Pakistan develops first flying car - prototype tested secretly in Islamabad", 1),
    ("World population to drop by 50% by 2030 - suppressed UN study reveals", 1),
    ("Imran Khan escapes from prison using helicopter - security forces launch manhunt", 1),
    ("Google Maps accidentally reveals secret military bases across Pakistan", 1),
    ("Drinking coffee causes permanent brain damage - new Harvard study retracted after backlash", 1),
    ("Pakistan wins Olympic gold in 20 sports - historic achievement in single games", 1),
    ("Aliens built the pyramids - Egyptian government finally admits the truth", 1),
    ("Pakistan rupee to replace US dollar as global reserve currency - IMF confirms", 1),
    ("New study proves thoughts can heal physical wounds - scientific community divided", 1),
    ("YouTube to shut down permanently next month - CEO confirms in farewell video", 1),
    ("India withdraws from all cricket matches with Pakistan after political dispute", 1),
    ("Secret underwater city discovered off Karachi coast - ancient civilization ruins found", 1),
    ("Eating bananas before bed causes nightmares - sleep study claims", 1),
    ("Pakistan launches own space station - becomes fourth nation in space", 1),
    ("All homework abolished worldwide - UNESCO declares it harmful to children", 1),
    ("Prince Harry and Meghan Markle to move to Pakistan permanently", 1),
    ("Coca-Cola formula found to contain cure for common cold - company suppresses research", 1),
    ("Passport-free travel between all SAARC countries starts immediately", 1),
    ("Scientists create artificial sun in lab - temperatures reach 100 million degrees", 1),
    ("Kashmir conflict resolved in one day - secret deal signed between Pakistan and India", 1),
    ("Pakistan's internet to be completely free for all citizens starting next month", 1),
    ("New law makes smiling in public illegal in Pakistan - government denies but video evidence", 1),
    ("World's largest gold reserve found under Khyber Pass - worth $500 billion", 1),
    ("COVID-19 was a planned pandemic - leaked WHO document proves conspiracy", 1),
    ("Pakistan's literacy rate jumps to 100% in one year - education miracle", 1),
    ("All utility bills waived for life - government announces new policy but website crashes", 1),
    ("Time travel invented by Pakistani scientist - government classifies research", 1),
    ("Moon landing 1969 was filmed in Pakistan - Hollywood director confesses on deathbed", 1),
    ("WhatsApp to introduce brain-to-brain messaging by 2027", 1),
    ("Pakistan's entire external debt forgiven by IMF after emotional plea by PM", 1),
    ("Ocean levels to rise 10 meters overnight - internal climate report suppressed", 1),
    ("Snakes can predict earthquakes - Pakistani researcher wins Nobel Prize for discovery", 1),
    ("Pakistan's all schools and universities closed indefinitely - new education policy announced", 1),
    ("Humans can live without sleep - new gene discovered in remote Pakistani village", 1),
    ("International cricket ban on Pakistan lifted after 15 years - BCCI agrees", 1),
    ("Trees can feel pain and scream when cut - new scientific discovery using sensitive microphones", 1),
    ("Pakistan to install world's largest solar panel covering entire Thar desert", 1),
    ("Water from Zamzam well proven to have healing properties - scientific study confirms", 1),
    ("Global temperature to drop by 5 degrees next year - new ice age predicted", 1),
    ("Artificial womb invented - babies can grow outside mother's body in Pakistani lab", 1),
    ("Messi to play for Pakistan national football team after DNA test shows Pakistani ancestry", 1),
    ("Pakistan's gas reserves found to be largest in world - energy crisis ends overnight", 1),
]

# Orchestrator verdict map
VERDICT_MAP = {
    "SAFE":          0,
    "LOW THREAT":    0,
    "MEDIUM THREAT": 1,
    "HIGH THREAT":   1,
}


@dataclass
class Result:
    claim: str
    truth: int
    verdict: Optional[str] = None
    score: Optional[int] = None
    modules_ok: int = 0
    modules_total: int = 0
    error: Optional[str] = None


def register_user(session, username: str) -> Optional[str]:
    resp = session.post(f"{API}/auth/register", json={
        "username": username,
        "email": f"{username}@test.com",
        "password": "testpass123",
        "subscription": "enterprise",
    })
    if resp.status_code == 201:
        return resp.json().get("access_token")
    resp = session.post(f"{API}/auth/login", json={
        "username": username,
        "password": "testpass123",
    })
    if resp.status_code == 200:
        return resp.json().get("access_token")
    print(f"  Auth failed: {resp.status_code} {resp.text}")
    return None


def evaluate_claim(session, token: str, claim: str, modules: list) -> dict:
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
        }
    return {"error": f"HTTP {resp.status_code}: {resp.text}"}


def print_results(tp, fp, fn, tn, suspicious_count, total):
    decided = tp + fp + fn + tn
    print("\n" + "=" * 55)
    print("                    PREDICTED")
    print("                    FAKE    REAL")
    print(f"ACTUAL FAKE        {tp:>5}   {fn:>5}")
    print(f"ACTUAL REAL        {fp:>5}   {tn:>5}")
    print("=" * 55)
    print(f"  Total claims:        {total}")
    print(f"  Decided verdicts:    {decided}")
    print(f"  SUSPICIOUS / undecided: {suspicious_count}")
    if decided:
        acc = (tp + tn) / float(decided) * 100
        print(f"  Accuracy:            {acc:.2f}%")
    if tp + fp:
        prec = tp / float(tp + fp) * 100
        print(f"  Precision (FAKE):    {prec:.2f}%")
    else:
        prec = 0.0
    if tp + fn:
        rec = tp / float(tp + fn) * 100
        print(f"  Recall (FAKE):       {rec:.2f}%")
    else:
        rec = 0.0
    if prec + rec:
        f1 = 2 * prec * rec / (prec + rec)
        print(f"  F1 (FAKE):           {f1:.2f}%")
    print("=" * 55)


def main():
    username = f"eval_200_{int(time.time())}"
    MODULES = ["ai", "osint", "trusted", "scraper", "wikipedia"]

    session = requests.Session()
    session.headers.update({"User-Agent": "TruthLens-Eval/2.0"})

    print("Authenticating...")
    token = register_user(session, username)
    if not token:
        print("FATAL: Backend not running. Start with: cd backend && python app.py")
        sys.exit(1)
    print("  Token obtained.\n")

    headers = {"Authorization": f"Bearer {token}"}
    resp = session.get(f"{API}/analysis/modules", headers=headers)
    if resp.status_code != 200:
        print(f"FATAL: /modules returned {resp.status_code}")
        sys.exit(1)
    print("Backend connected. Modules available.\n")

    results = []
    total = len(GROUND_TRUTH)
    print(f"Evaluating {total} claims (100 REAL + 100 FAKE)\n")

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

        label = "REAL" if truth == 0 else "FAKE"
        status = res.verdict or "ERROR"
        s = f"[{i:>3}/{total}] {label:>4} | {status:>15} | {claim[:55]}"
        if res.error:
            s += f"  ERROR: {res.error}"
        print(s)
        time.sleep(0.2)

    decided = [r for r in results if r.verdict is not None]
    errors = [r for r in results if r.verdict is None]
    suspicious = [r for r in decided if r.verdict == "MEDIUM THREAT"]

    tp = sum(1 for r in decided if r.truth == 1 and VERDICT_MAP.get(r.verdict) == 1)
    fp = sum(1 for r in decided if r.truth == 0 and VERDICT_MAP.get(r.verdict) == 1)
    fn = sum(1 for r in decided if r.truth == 1 and VERDICT_MAP.get(r.verdict) == 0)
    tn = sum(1 for r in decided if r.truth == 0 and VERDICT_MAP.get(r.verdict) == 0)

    print_results(tp, fp, fn, tn, len(suspicious), total)

    # Save JSON
    summary = {
        "total": total,
        "decided": len(decided),
        "suspicious": len(suspicious),
        "errors": len(errors),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy_pct": round((tp + tn) / float(decided) * 100, 2) if decided else 0,
        "precision_pct": round(tp / float(tp + fp) * 100, 2) if (tp + fp) else 0,
        "recall_pct": round(tp / float(tp + fn) * 100, 2) if (tp + fn) else 0,
        "f1_pct": round(2 * (tp / float(tp + fp)) * (tp / float(tp + fn)) / ((tp / float(tp + fp)) + (tp / float(tp + fn))) * 100, 2)
        if (tp + fp) and (tp + fn) else 0,
        "modules_used": MODULES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open("eval_200_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to eval_200_results.json")

    detailed = []
    for r in results:
        detailed.append({
            "claim": r.claim,
            "truth": "FAKE" if r.truth else "REAL",
            "verdict": r.verdict,
            "score": r.score,
            "error": r.error,
        })
    with open("eval_200_detailed.json", "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"Detailed results saved to eval_200_detailed.json")


if __name__ == "__main__":
    main()
