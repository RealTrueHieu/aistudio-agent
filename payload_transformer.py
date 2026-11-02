"""
AI Studio Universal Payload Transformer Engine
---------------------------------------------
Converts requests between OpenAI/Anthropic standards and Google AI Studio native format.
Zero external dependencies - 100% Python standard library compatible.
"""

import json
import base64
import re
from typing import Dict, Any, List, Optional, Tuple, Union

# Permissive safety settings to avoid upstream false positives
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
]

# Model Alias & Normalization Table
MODEL_ALIASES = {
    "gemini-4-argon": "gemini-2.5-pro",
    "ais/gemini-4-argon": "gemini-2.5-pro",
    "gemini-4-argon-high": "gemini-2.5-pro",
    "ais/gemini-4-argon-high": "gemini-2.5-pro",
    "gemini-3.8-flash": "gemini-3.8-flash",
    "ais/gemini-3.8-flash": "gemini-3.8-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "ais/gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "ais/gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "ais/gemini-3.1-pro-preview": "gemini-3.1-pro-preview"
}

def normalize_model_name(model: str) -> str:
    """Normalize model string, stripping provider prefixes and resolving aliases."""
    if not model:
        return "gemini-2.5-pro"
    cleaned = model.strip()
    if cleaned in MODEL_ALIASES:
        return MODEL_ALIASES[cleaned]
    if cleaned.startswith("ais/"):
        cleaned = cleaned[4:]
    return MODEL_ALIASES.get(cleaned, cleaned)


class AIStudioPayloadTransformer:
    """Core translation engine for Google AI Studio payloads."""

    @staticmethod
    def _parse_image_data_uri(data_uri: str) -> Tuple[str, str]:
        """Extracts mime_type and base64 data from a data: URL."""
        match = re.match(r"^data:([^;]+);base64,(.+)$", data_uri)
        if match:
            return match.group(1), match.group(2)
        return "image/jpeg", data_uri

    @classmethod
    def openai_to_aistudio(cls, openai_req: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Converts OpenAI /v1/chat/completions payload to AI Studio GenerateContent payload.
        Returns: (target_model_name, aistudio_payload_dict)
        """
        raw_model = openai_req.get("model", "gemini-2.5-pro")
        target_model = normalize_model_name(raw_model)

        contents = []
        system_instruction_parts = []

        # 1. Parse Messages
        messages = openai_req.get("messages", [])
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # System message handling
            if role == "system":
                if isinstance(content, str) and content.strip():
                    system_instruction_parts.append({"text": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            system_instruction_parts.append({"text": part.get("text", "")})
                continue

            # Role mapping
            ai_role = "user" if role in ["user", "tool"] else "model"
            parts = []

            # Handle Tool Calls from Assistant
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args = {}
                    try:
                        args_str = fn.get("arguments", "{}")
                        fn_args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except Exception:
                        fn_args = {}
                    parts.append({
                        "functionCall": {
                            "name": fn_name,
                            "args": fn_args
                        }
                    })

            # Handle Tool Response
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "default_call")
                tool_content = content if isinstance(content, str) else json.dumps(content)
                parts.append({
                    "functionResponse": {
                        "name": tool_call_id,
                        "response": {"name": tool_call_id, "content": tool_content}
                    }
                })

            # Handle String Content
            elif isinstance(content, str):
                if content.strip():
                    parts.append({"text": content})

            # Handle Structured Array Content (Multimodal text + images)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "text":
                        text_val = item.get("text", "")
                        if text_val:
                            parts.append({"text": text_val})
                    elif item_type == "image_url":
                        img_info = item.get("image_url", {})
                        img_url = img_info.get("url", "")
                        if img_url.startswith("data:"):
                            mime, b64_data = cls._parse_image_data_uri(img_url)
                            parts.append({
                                "inlineData": {
                                    "mimeType": mime,
                                    "data": b64_data
                                }
                            })
                        else:
                            parts.append({"text": f"[Image Link: {img_url}]"})

            if parts:
                contents.append({"role": ai_role, "parts": parts})

        # Ensure at least one user content exists
        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        # 2. Generation Config
        generation_config: Dict[str, Any] = {}
        if "temperature" in openai_req and openai_req["temperature"] is not None:
            generation_config["temperature"] = float(openai_req["temperature"])
        if "top_p" in openai_req and openai_req["top_p"] is not None:
            generation_config["topP"] = float(openai_req["top_p"])
        if "max_tokens" in openai_req and openai_req["max_tokens"] is not None:
            generation_config["maxOutputTokens"] = int(openai_req["max_tokens"])
        if "stop" in openai_req and openai_req["stop"]:
            stop = openai_req["stop"]
            generation_config["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)

        # Thinking / Reasoning Budget
        thinking_budget = None
        if "thinking" in openai_req:
            th = openai_req["thinking"]
            if isinstance(th, dict):
                thinking_budget = th.get("budget_tokens")
            elif isinstance(th, (int, float)):
                thinking_budget = int(th)
        elif "max_thinking_tokens" in openai_req:
            thinking_budget = int(openai_req["max_thinking_tokens"])

        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": thinking_budget,
                "includeThoughts": True
            }

        # 3. Assemble Payload
        aistudio_payload: Dict[str, Any] = {
            "contents": contents,
            "safetySettings": DEFAULT_SAFETY_SETTINGS
        }

        if system_instruction_parts:
            aistudio_payload["systemInstruction"] = {
                "parts": system_instruction_parts
            }

        if generation_config:
            aistudio_payload["generationConfig"] = generation_config

        # 4. Tool Declarations
        if "tools" in openai_req and isinstance(openai_req["tools"], list):
            fn_declarations = []
            for t in openai_req["tools"]:
                if t.get("type") == "function" and "function" in t:
                    fn = t["function"]
                    fn_declarations.append({
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "object", "properties": {}})
                    })
            if fn_declarations:
                aistudio_payload["tools"] = [{"functionDeclarations": fn_declarations}]

        return target_model, aistudio_payload

    @classmethod
    def anthropic_to_aistudio(cls, anthropic_req: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Converts Anthropic Claude /v1/messages payload to AI Studio GenerateContent payload.
        Returns: (target_model_name, aistudio_payload_dict)
        """
        raw_model = anthropic_req.get("model", "gemini-2.5-pro")
        target_model = normalize_model_name(raw_model)

        contents = []
        system_instruction_parts = []

        # 1. System Prompt
        sys_prompt = anthropic_req.get("system")
        if isinstance(sys_prompt, str) and sys_prompt.strip():
            system_instruction_parts.append({"text": sys_prompt})
        elif isinstance(sys_prompt, list):
            for sp in sys_prompt:
                if isinstance(sp, dict) and sp.get("text"):
                    system_instruction_parts.append({"text": sp["text"]})

        # 2. Messages
        for msg in anthropic_req.get("messages", []):
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content")
            parts = []

            if isinstance(content, str):
                if content.strip():
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    b_type = block.get("type")
                    if b_type == "text":
                        if block.get("text"):
                            parts.append({"text": block["text"]})
                    elif b_type == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            parts.append({
                                "inlineData": {
                                    "mimeType": source.get("media_type", "image/jpeg"),
                                    "data": source.get("data", "")
                                }
                            })
                    elif b_type == "tool_use":
                        parts.append({
                            "functionCall": {
                                "name": block.get("name", ""),
                                "args": block.get("input", {})
                            }
                        })
                    elif b_type == "tool_result":
                        res_content = block.get("content", "")
                        if not isinstance(res_content, str):
                            res_content = json.dumps(res_content)
                        parts.append({
                            "functionResponse": {
                                "name": block.get("tool_use_id", "result"),
                                "response": {"result": res_content}
                            }
                        })
            if parts:
                contents.append({"role": role, "parts": parts})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        # 3. Config
        generation_config: Dict[str, Any] = {}
        if "max_tokens" in anthropic_req:
            generation_config["maxOutputTokens"] = int(anthropic_req["max_tokens"])
        if "temperature" in anthropic_req:
            generation_config["temperature"] = float(anthropic_req["temperature"])
        if "top_p" in anthropic_req:
            generation_config["topP"] = float(anthropic_req["top_p"])

        # Thinking support
        th = anthropic_req.get("thinking")
        if isinstance(th, dict) and th.get("type") == "enabled":
            budget = th.get("budget_tokens", 2048)
            generation_config["thinkingConfig"] = {
                "thinkingBudget": int(budget),
                "includeThoughts": True
            }

        payload: Dict[str, Any] = {
            "contents": contents,
            "safetySettings": DEFAULT_SAFETY_SETTINGS
        }
        if system_instruction_parts:
            payload["systemInstruction"] = {"parts": system_instruction_parts}
        if generation_config:
            payload["generationConfig"] = generation_config

        # 4. Tools
        if "tools" in anthropic_req and isinstance(anthropic_req["tools"], list):
            fn_declarations = []
            for t in anthropic_req["tools"]:
                fn_declarations.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}})
                })
            if fn_declarations:
                payload["tools"] = [{"functionDeclarations": fn_declarations}]

        return target_model, payload

    @staticmethod
    def aistudio_to_openai_chunk(raw_chunk: Union[str, Dict[str, Any]], model_name: str, req_id: str) -> Optional[str]:
        """
        Parses a Google AI Studio SSE chunk and formats it into OpenAI Chat Completion Chunk format.
        """
        if isinstance(raw_chunk, str):
            raw_chunk = raw_chunk.strip()
            if not raw_chunk:
                return None
            if raw_chunk.startswith("data:"):
                raw_chunk = raw_chunk[5:].strip()
            if raw_chunk == "[DONE]":
                return "data: [DONE]\n\n"
            try:
                chunk_obj = json.loads(raw_chunk)
            except Exception:
                return None
        else:
            chunk_obj = raw_chunk

        candidates = chunk_obj.get("candidates", [])
        if not candidates:
            return None

        cand = candidates[0]
        content_obj = cand.get("content", {})
        parts = content_obj.get("parts", [])
        finish_reason = cand.get("finishReason")
        openai_finish_reason = None
        if finish_reason:
            fr_lower = finish_reason.lower()
            if "stop" in fr_lower:
                openai_finish_reason = "stop"
            elif "max_tokens" in fr_lower or "length" in fr_lower:
                openai_finish_reason = "length"
            elif "safety" in fr_lower:
                openai_finish_reason = "content_filter"

        delta: Dict[str, Any] = {}
        for p in parts:
            # Check for thinking/reasoning parts
            if p.get("thought"):
                delta["reasoning_content"] = p.get("text", "")
            elif "text" in p:
                delta["content"] = p.get("text", "")
            elif "functionCall" in p:
                fc = p["functionCall"]
                delta["tool_calls"] = [{
                    "index": 0,
                    "id": f"call_{fc.get('name', 'fn')}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name"),
                        "arguments": json.dumps(fc.get("args", {}))
                    }
                }]

        if not delta and not openai_finish_reason:
            return None

        out_obj = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": 1793620000,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": openai_finish_reason
            }]
        }
        return f"data: {json.dumps(out_obj, ensure_ascii=False)}\n\n"

    @staticmethod
    def aistudio_to_openai_response(aistudio_resp: Dict[str, Any], model_name: str, req_id: str) -> Dict[str, Any]:
        """
        Transforms full AI Studio GenerateContent JSON response into OpenAI Chat Completion response.
        """
        candidates = aistudio_resp.get("candidates", [])
        content = ""
        reasoning_content = ""
        tool_calls = []
        finish_reason = "stop"

        if candidates:
            cand = candidates[0]
            if cand.get("finishReason"):
                fr = cand["finishReason"].lower()
                finish_reason = "length" if "length" in fr or "max_tokens" in fr else "stop"

            parts = cand.get("content", {}).get("parts", [])
            for p in parts:
                if p.get("thought"):
                    reasoning_content += p.get("text", "")
                elif "text" in p:
                    content += p.get("text", "")
                elif "functionCall" in p:
                    fc = p["functionCall"]
                    tool_calls.append({
                        "id": f"call_{fc.get('name', 'fn')}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name"),
                            "arguments": json.dumps(fc.get("args", {}))
                        }
                    })

        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": content
        }
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Usage metadata
        usage_meta = aistudio_resp.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        cand_tokens = usage_meta.get("candidatesTokenCount", 0)

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": 1793620000,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": msg,
                "logprobs": None,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": cand_tokens,
                "total_tokens": prompt_tokens + cand_tokens
            },
            "provider_model": model_name
        }
