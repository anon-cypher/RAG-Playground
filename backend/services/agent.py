"""Agentic RAG loop using OpenRouter LLM."""
from openai import OpenAI

def run_agentic_loop(
    query: str, 
    initial_context: list[str], 
    max_steps: int = 3, 
    openrouter_api_key: str = None, 
    llm_model: str = None,
    retriever_func = None
) -> tuple[str, list[dict], dict]:
    """
    Run an Agentic loop. 
    The LLM can output [SEARCH: <query>] to pull more chunks or [SYNTHESIZE: <answer>] to finish.
    """
    if not openrouter_api_key:
        return "Agentic search requires OpenRouter API Key.", [], {}
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    
    system_prompt = (
        "You are an advanced iterative RAG agent. You must either SEARCH for more info, or SYNTHESIZE an answer.\\n"
        "IF you need more information to answer the user's query, reply EXACTLY with: [SEARCH: your search query]\\n"
        "IF the provided context has enough information, reply EXACTLY with: [SYNTHESIZE: your final comprehensive answer]\\n"
        "Do nothing else. Only output [SEARCH: ...] or [SYNTHESIZE: ...]."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Initial Context: {' '.join(initial_context)}\\n\\nUser Query: {query}"}
    ]
    
    logs = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for step in range(max_steps):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0.2,
                max_tokens=800
            ) # wait, prompt uses \\n because I escaped it in write_to_file, wait it's raw string here so \\n is literal backslash n. I'll just use \n below directly in python strings. But wait, JSON string parsing handles \n safely. Oh, I escaped it. It will be 2 backslashes. That's fine for OpenAI to read.
            content = response.choices[0].message.content.strip()
        except Exception as e:
            return f"Agent Error: {str(e)}", logs, total_usage
            
        logs.append({"step": step+1, "action": content})
        messages.append({"role": "assistant", "content": content})
        
        if content.startswith("[SYNTHESIZE:"):
            return content.replace("[SYNTHESIZE:", "").strip("] "), logs, total_usage
        elif content.startswith("[SEARCH:"):
            search_query = content.replace("[SEARCH:", "").strip("] ")
            if retriever_func:
                new_context = retriever_func(search_query)
                messages.append({"role": "user", "content": f"New Context retrieved for '{search_query}': {' '.join(new_context)}"})
            else:
                messages.append({"role": "user", "content": "Search failed (No retriever provided)."})
        else:
            if step == max_steps - 1:
                return content, logs, total_usage
            messages.append({"role": "user", "content": "Please format your output strictly as [SEARCH: query] or [SYNTHESIZE: answer]."})

    return "Agent reached max steps without synthesis.", logs, total_usage
