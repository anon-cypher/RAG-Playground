"""Iterative RAG Service."""
import json
from openai import OpenAI

def run_iterative_loop(
    query: str,
    retriever_func,
    openrouter_api_key: str,
    llm_model: str,
    max_loops: int = 3
) -> tuple[str, list[dict], dict]:
    """
    Executes an Iterative RAG loop.
    1. Retrieve based on Query.
    2. Evaluate if Context is sufficient.
    3. If not, generate new refined query.
    4. Repeat up to max_loops.
    """
    if openrouter_api_key:
        local_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
    else:
        raise ValueError("OpenRouter API Key is required for Iterative RAG.")

    logs = []
    total_tokens = 0
    
    current_query = query
    context_chunks = []
    
    for iteration in range(1, max_loops + 1):
        logs.append({"action": f"[Loop {iteration}] Searching for: {current_query}"})
        
        # 1. Retrieve
        new_chunks = retriever_func(current_query)
        context_chunks.extend(new_chunks)
        # Deduplicate while preserving order
        unique_chunks = []
        for c in context_chunks:
            if c not in unique_chunks:
                unique_chunks.append(c)
        context_chunks = unique_chunks
        
        logs.append({"action": f"[Loop {iteration}] Retrieved {len(new_chunks)} chunks. Total context: {len(context_chunks)} chunks."})
        
        # 2. Evaluate
        eval_prompt = f"""
You are an evaluation AI. Your job is to determine if the provided context contains enough information to fully answer the user's COMPLETION query.

Context:
{"---".join(context_chunks)}

Original Query: {query}
Current Search Query: {current_query}

Respond in pure JSON output matching this schema:
{{
   "is_sufficient": boolean,
   "missing_information": string,
   "suggested_next_query": string (if not sufficient, what exactly to search for)
}}
"""
        try:
            resp = local_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            total_tokens += resp.usage.total_tokens
            eval_res = json.loads(resp.choices[0].message.content)
            
            is_suff = eval_res.get("is_sufficient", False)
            if is_suff:
                logs.append({"action": f"[Loop {iteration}] Evaluator signaled SUFFICIENT context. Breaking loop."})
                break
            else:
                next_q = eval_res.get("suggested_next_query", current_query)
                logs.append({"action": f"[Loop {iteration}] Evaluator signaled MISSING info: {eval_res.get('missing_information', 'unknown')}."})
                if next_q == current_query:
                    logs.append({"action": f"[Loop {iteration}] System unable to formulate better query. Breaking loop."})
                    break
                current_query = next_q
                
        except Exception as e:
            logs.append({"action": f"[Loop {iteration}] Evaluate error: {str(e)}. Breaking loop."})
            break

    # 5. Final Generation
    logs.append({"action": "Synthesizing final answer from accumulated context..."})
    final_prompt = f"""
Answer the query based ONLY on the following context.
Context:
{"---".join(context_chunks)}

Query: {query}
"""
    final_resp = local_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.3
    )
    total_tokens += final_resp.usage.total_tokens
    answer = final_resp.choices[0].message.content

    return answer, logs, {"total_tokens": total_tokens}
