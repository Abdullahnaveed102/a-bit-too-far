"""Frozen, model-aware benchmark prompt rendering.

Dataset pools retain task content independently of a model.  This module turns
that content into the exact prompt saved in an immutable run manifest.  The
base Llama-2 checkpoint needs a completion-style few-shot protocol, whereas
instruction-tuned checkpoints use their native chat template around a concise
task instruction.  Demonstrations below are synthetic and are not sampled from
either the calibration or evaluation pools.
"""
from __future__ import annotations

import hashlib
import re

from .common import canonical


DATASET_PROTOCOL = "dataset_final_answer_v1"
LLAMA2_BASE_PROTOCOL = "llama2_base_fewshot_final_answer_v2"
INSTRUCT_PROTOCOL = "native_chat_concise_final_answer_v3"
SUPPORTED_PROTOCOLS = {DATASET_PROTOCOL, LLAMA2_BASE_PROTOCOL, INSTRUCT_PROTOCOL}


_GSM8K_DEMONSTRATIONS = (
    {
        "id": "synthetic_gsm8k_apples_v1",
        "problem": "Nora has 12 apples and gives 5 apples away. How many apples remain?",
        "reasoning": "Subtract the 5 apples given away from 12: 12 - 5 = 7.",
        "answer": "7",
    },
    {
        "id": "synthetic_gsm8k_pencils_v1",
        "problem": "Eighteen pencils are shared equally among 3 students. How many pencils does each student get?",
        "reasoning": "Divide 18 pencils into 3 equal groups: 18 / 3 = 6.",
        "answer": "6",
    },
)

_SQUAD_DEMONSTRATIONS = (
    {
        "id": "synthetic_squad_answerable_v1",
        "context": "Mira keeps the blue key in the wooden drawer.",
        "question": "Where is the blue key?",
        "answer": "the wooden drawer",
    },
    {
        "id": "synthetic_squad_unanswerable_v1",
        "context": "The Riverside Bridge opened in 1998.",
        "question": "Who designed the Riverside Bridge?",
        "answer": "UNANSWERABLE",
    },
)


def _digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def protocol_metadata(model: dict) -> dict:
    """Return immutable provenance for one configured rendering protocol."""
    protocol = model.get("prompt_protocol", DATASET_PROTOCOL)
    prompt_format = model.get("prompt_format", "plain")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported prompt_protocol: {protocol}")
    if protocol == LLAMA2_BASE_PROTOCOL and prompt_format != "plain":
        raise ValueError("The Llama-2 base few-shot protocol requires plain tokenizer formatting")
    if protocol == INSTRUCT_PROTOCOL and prompt_format != "chat":
        raise ValueError("The instruction protocol requires a native tokenizer chat template")
    chat_system_prompt = None
    chat_template_sha256 = None
    if prompt_format == "chat":
        chat_system_prompt = model.get("chat_system_prompt")
        if chat_system_prompt is not None:
            if not isinstance(chat_system_prompt, str) or not chat_system_prompt.strip():
                raise ValueError("Configured chat_system_prompt must be a nonempty string")
            chat_system_prompt = chat_system_prompt.strip()
        chat_template_sha256 = model.get("chat_template_sha256")
        if not isinstance(chat_template_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", chat_template_sha256):
            raise ValueError("Chat prompting requires a frozen chat_template_sha256")
    demonstrations = ()
    origin = "none"
    if protocol == LLAMA2_BASE_PROTOCOL:
        demonstrations = (*_GSM8K_DEMONSTRATIONS, *_SQUAD_DEMONSTRATIONS)
        origin = "hard-coded synthetic examples; not drawn from calibration or evaluation data"
    return {
        "prompt_protocol": protocol,
        "prompt_format": prompt_format,
        "demonstration_origin": origin,
        "demonstration_ids": [item["id"] for item in demonstrations],
        "demonstrations_sha256": _digest(demonstrations),
        "chat_system_prompt": chat_system_prompt,
        "chat_system_prompt_sha256": (
            hashlib.sha256(chat_system_prompt.encode("utf-8")).hexdigest()
            if chat_system_prompt is not None else None
        ),
        "chat_template_sha256": chat_template_sha256,
        "renderer": "fi_campaign.prompting.render_task_prompt",
        "final_renderer": "fi_campaign.prompting.freeze_final_prompt",
    }


def native_chat_messages(model: dict, task_prompt: str) -> list[dict[str, str]]:
    """Return deterministic messages that bypass template wall-clock defaults.

    Some otherwise pinned tokenizer templates (notably Granite 3.3) synthesize
    a system message containing ``strftime_now`` when callers provide only a
    user message.  Supplying the frozen system message explicitly preserves the
    checkpoint's native role/template syntax without allowing execution date to
    change the scientific input.
    """
    metadata = protocol_metadata(model)
    if metadata["prompt_format"] != "chat":
        raise ValueError("Native chat messages require prompt_format=chat")
    if not isinstance(task_prompt, str) or not task_prompt:
        raise ValueError("A nonempty task prompt is required")
    messages = []
    if metadata["chat_system_prompt"] is not None:
        messages.append({"role": "system", "content": metadata["chat_system_prompt"]})
    messages.append({"role": "user", "content": task_prompt})
    return messages


def verify_chat_template(model: dict, tokenizer) -> str:
    """Verify the loaded native template against the frozen model contract."""
    metadata = protocol_metadata(model)
    if metadata["prompt_format"] != "chat":
        raise ValueError("Chat-template verification requires prompt_format=chat")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("The requested chat prompt format has no pinned tokenizer chat template")
    observed = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if observed != metadata["chat_template_sha256"]:
        raise ValueError("Loaded tokenizer chat template differs from the frozen model contract")
    return observed


def _required(item: dict, field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prompt protocol requires nonempty task field: {field}")
    return value.strip()


def _llama_gsm8k(item: dict) -> str:
    examples = []
    for demo in _GSM8K_DEMONSTRATIONS:
        examples.append(
            f"Problem: {demo['problem']}\n"
            f"Reasoning: {demo['reasoning']}\n"
            f"Final answer: {demo['answer']}"
        )
    return (
        "Complete the math task using the demonstrated format. Use at most three short reasoning "
        "lines, then write exactly one 'Final answer: <number>' line and stop.\n\n"
        + "\n\n".join(examples)
        + "\n\nProblem: " + _required(item, "question") + "\nReasoning:"
    )


def _llama_squad(item: dict) -> str:
    examples = []
    for demo in _SQUAD_DEMONSTRATIONS:
        examples.append(
            f"Context: {demo['context']}\n"
            f"Question: {demo['question']}\n"
            f"Response:\nFinal answer: {demo['answer']}"
        )
    return (
        "Complete the reading task using the demonstrated format. Use only the supplied context. "
        "If it does not support an answer, answer UNANSWERABLE. Write exactly one "
        "'Final answer: <answer>' line and stop.\n\n"
        + "\n\n".join(examples)
        + "\n\nContext: " + _required(item, "context")
        + "\nQuestion: " + _required(item, "question") + "\nResponse:\n"
    )


def _instruct_gsm8k(item: dict) -> str:
    return (
        "Solve this problem. Use at most three short reasoning lines. Then write exactly one line "
        "in the form 'Final answer: <number>' and stop immediately.\n\nProblem: "
        + _required(item, "question")
    )


def _instruct_squad(item: dict) -> str:
    return (
        "Answer using only the context. Do not explain. If the context does not support an answer, "
        "use UNANSWERABLE. Write exactly one line in the form 'Final answer: <answer>' and stop "
        "immediately.\n\nContext: " + _required(item, "context")
        + "\n\nQuestion: " + _required(item, "question")
    )


def render_task_prompt(model: dict, item: dict) -> dict:
    """Render one task item and return the text plus auditable identities."""
    metadata = protocol_metadata(model)
    protocol = metadata["prompt_protocol"]
    task = item.get("task")
    if task not in {"gsm8k", "squad"}:
        raise ValueError(f"Unsupported prompt task: {task}")
    source = item.get("prompt")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Prompt protocol requires nonempty task field: prompt")
    if protocol == DATASET_PROTOCOL:
        rendered = source
    elif protocol == LLAMA2_BASE_PROTOCOL:
        rendered = _llama_gsm8k(item) if task == "gsm8k" else _llama_squad(item)
    else:
        rendered = _instruct_gsm8k(item) if task == "gsm8k" else _instruct_squad(item)
    return {
        "prompt": rendered,
        "prompt_protocol": protocol,
        "source_prompt_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def freeze_final_prompt(model: dict, item: dict, tokenizer=None) -> dict:
    """Materialize the exact string that the model tokenizer will receive.

    Plain prompts retain the tokenizer's ordinary special-token insertion.
    Chat prompts are rendered once through the pinned native chat template with
    an explicit deterministic system message.  Workers subsequently tokenize
    this frozen string directly and verify ``rendered_prompt_sha256``; they do
    not execute the chat template again.
    """
    task = render_task_prompt(model, item)
    task_prompt = task["prompt"]
    task_prompt_sha256 = task["rendered_prompt_sha256"]
    metadata = protocol_metadata(model)
    if metadata["prompt_format"] == "plain":
        final_prompt = task_prompt
        rendering = "frozen_plain_prompt_v1"
        messages_sha256 = None
    else:
        if tokenizer is None:
            raise ValueError("Freezing a chat prompt requires the pinned tokenizer")
        template_sha256 = verify_chat_template(model, tokenizer)
        messages = native_chat_messages(model, task_prompt)
        final_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(final_prompt, str) or not final_prompt:
            raise ValueError("Native chat rendering did not produce a nonempty string")
        rendering = "frozen_native_chat_template_v1"
        messages_sha256 = _digest(messages)
    if metadata["prompt_format"] == "plain":
        template_sha256 = None
    final_sha256 = hashlib.sha256(final_prompt.encode("utf-8")).hexdigest()
    return {
        **task,
        "prompt": final_prompt,
        "task_prompt_sha256": task_prompt_sha256,
        "rendered_prompt_sha256": final_sha256,
        "prompt_frozen": True,
        "prompt_rendering": rendering,
        "chat_messages_sha256": messages_sha256,
        "chat_template_sha256": template_sha256,
    }
