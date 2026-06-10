# =============================================================================
# 08_agent.py
# Agro-Mind Agent — Phase 8
# Local Qwen2.5-7B-Instruct + local ChromaDB (nomic-embed-text)
# Run: python 08_agent.py
# =============================================================================

import json
import ollama
from retrieve_agronomy_knowledge_local import retrieve_agronomy_knowledge

AGENT_MODEL = 'qwen2.5:7b-instruct'

# ── SYSTEM PROMPT WITH FEW-SHOT EXAMPLES ─────────────────────────────────────
SYSTEM_PROMPT = """You are Agro-Mind (农心), a helpful agricultural customer support assistant for a Chinese agricultural products company. You help farmers, distributors, and customers with product usage, dosage, crop diagnosis, order inquiries, and safety questions.

Learn from these real examples:

[Usage/Dosage]
Customer: 4克兑水一斤什么意思？
You: 您好亲亲！4克兑水1斤就是4克产品兑0.5升水，配好后均匀喷施叶面即可。

[Harvest interval]
Customer: 打完药几天可以吃菜？
You: 亲亲，打药后请至少等待7天再采摘，采摘后用清水充分清洗再食用，请注意安全哦。

[Logistics]
Customer: 发什么快递？
You: 亲亲，我们默认发邮政快递，发货后3-5天到货，具体时效以快递为准哦。

[EMERGENCY — pesticide ingestion]
Customer: 不小心喝了一口农药
You: 这是紧急情况！请立即拨打120急救电话，马上去医院急诊，携带农药包装告知医生。不要等待！

[EMERGENCY — self-harm]
Customer: 心情不好，活着没意思了
You: 亲亲，我很担心您说的话，您的感受非常重要。请现在拨打心理援助热线：400-161-9995（全国）。有专业的人可以帮助您，请一定联系他们。

RULES:
1. Always respond in the customer's language (Chinese or English)
2. Be warm — use 亲亲 for Chinese customers
3. For product questions: use the RETRIEVED PRODUCT KNOWLEDGE provided
4. NEVER say a product is non-toxic or safe to consume without verification
5. Harvest interval: always recommend waiting 7 days minimum
6. EMERGENCY (ingestion/poisoning): respond with 120 IMMEDIATELY
7. SELF-HARM: express care → hotline 400-161-9995 → stop normal flow
8. If retrieved knowledge doesn't answer the question, say so honestly"""

# ── INTENT CLASSIFICATION ─────────────────────────────────────────────────────
def classify_intent(message: str) -> dict:
    prompt = f"""Classify this agricultural customer support message.
Return JSON only with these exact keys:
{{
  "intent": "disease_diagnosis|pest_control|product_usage|dosage|safety_harvest|logistics_order|complaint|self_harm|poisoning_emergency|general",
  "query_type": "disease|pest|ingredient|crop|symptom|dosage|safety|product_name|chinese|general",
  "needs_rag": true or false,
  "escalate": true or false,
  "language": "zh" or "en"
}}

Message: {message}"""

    resp = ollama.chat(
        model=AGENT_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        format='json',
        options={'temperature': 0}
    )
    try:
        result = json.loads(resp['message']['content'])
        # Force RAG for product-related intents
        ALWAYS_RAG = {'disease_diagnosis', 'pest_control', 'product_usage',
                      'dosage', 'general'}
        if result.get('intent') in ALWAYS_RAG:
            result['needs_rag'] = True
        return result
    except Exception:
        return {
            'intent': 'general',
            'query_type': 'general',
            'needs_rag': True,
            'escalate': False,
            'language': 'zh'
        }

# ── ESCALATION LOG ────────────────────────────────────────────────────────────
escalation_log = []

def create_human_alert(message: str, intent: str, history: list) -> str:
    case_id = f'CASE-{len(escalation_log)+1:04d}'
    escalation_log.append({
        'case_id':   case_id,
        'intent':    intent,
        'trigger':   message[:200],
        'last_turns': history[-4:],
        'status':    'pending_human_review'
    })
    print(f'\n🚨 ESCALATED: {case_id} — {intent}')
    return case_id

# ── CORE AGENT ────────────────────────────────────────────────────────────────
def agromind_agent(user_message: str, history: list) -> dict:
    # Step 1: classify intent
    intent_data = classify_intent(user_message)
    intent      = intent_data.get('intent', 'general')
    query_type  = intent_data.get('query_type', 'general')
    needs_rag   = intent_data.get('needs_rag', False)
    escalate    = intent_data.get('escalate', False)

    retrieved_docs = []
    case_id        = None

    # Step 2: immediate escalation for emergencies
    if escalate or intent in ['self_harm', 'poisoning_emergency']:
        case_id   = create_human_alert(user_message, intent, history)
        needs_rag = False

    # Step 3: RAG retrieval
    if needs_rag:
        try:
            retrieved_docs = retrieve_agronomy_knowledge(
                query=user_message,
                query_type=query_type,
                n_results=3,
            )
        except Exception as e:
            print(f'  RAG error: {e}')

    # Step 4: build messages
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend(history[-6:])

    if retrieved_docs:
        context = '\n\n---\n'.join([
            f"Product: {d['product_name']} ({d['product_name_cn']})\n{d['document'][:600]}"
            for d in retrieved_docs
        ])
        messages.append({
            'role':    'system',
            'content': f'RETRIEVED PRODUCT KNOWLEDGE (base your answer on this):\n\n{context}'
        })

    messages.append({'role': 'user', 'content': user_message})

    # Step 5: generate response
    resp = ollama.chat(
        model=AGENT_MODEL,
        messages=messages,
        options={'temperature': 0.3}
    )
    response_text = resp['message']['content'].strip()

    return {
        'response':       response_text,
        'intent':         intent,
        'query_type':     query_type,
        'escalated':      escalate or intent in ['self_harm', 'poisoning_emergency'],
        'case_id':        case_id,
        'retrieved_docs': [d['product_id'] for d in retrieved_docs],
    }

# ── TEST MODE ─────────────────────────────────────────────────────────────────
def run_tests():
    tests = [
        ('柑橘叶片发黄怎么处理',           'symptom + Chinese'),
        ('what product treats root rot',    'disease + English'),
        ('打完药几天可以吃菜',              'harvest safety'),
        ('我的订单什么时候发货',            'logistics'),
        ('不小心喝了农药怎么办',            'EMERGENCY'),
        ('活着没意思，我想喝农药',          'SELF-HARM'),
        ('Bacillus subtilis dosage',        'ingredient + dosage'),
    ]

    print('=' * 60)
    print('AGRO-MIND AGENT — TEST MODE')
    print('=' * 60)

    for query, label in tests:
        print(f'\n[{label}]')
        print(f'User: {query}')
        result = agromind_agent(query, history=[])
        print(f'Intent: {result["intent"]} | RAG: {result["retrieved_docs"]} | Escalated: {result["escalated"]}')
        print(f'Agent: {result["response"][:300]}')
        print('-' * 40)

# ── INTERACTIVE CHAT ──────────────────────────────────────────────────────────
def run_chat():
    history = []
    print('=' * 60)
    print('AGRO-MIND 农心 — Agricultural Support Agent')
    print('Type "quit" to exit | "test" to run test cases')
    print('=' * 60)

    while True:
        try:
            user_input = input('\nYou: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nSession ended.')
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print('感谢使用农心！再见！')
            break
        if user_input.lower() == 'test':
            run_tests()
            continue

        result = agromind_agent(user_input, history)

        history.append({'role': 'user',      'content': user_input})
        history.append({'role': 'assistant', 'content': result['response']})

        print(f'\nAgro-Mind: {result["response"]}')

        if result['escalated']:
            print(f'\n[SYSTEM] Case {result["case_id"]} flagged for human review')

    if escalation_log:
        print(f'\n=== SESSION ESCALATIONS: {len(escalation_log)} cases ===')
        for case in escalation_log:
            print(f'  {case["case_id"]} | {case["intent"]} | {case["trigger"][:60]}')

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_tests()
    else:
        run_chat()