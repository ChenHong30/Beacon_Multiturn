# metrics.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for JSON-based evaluation (no CSV / Pandas)

import json
from typing import Any, Dict, List

import ifeval


# ============================================================
# 原封不动复用 Meta / IFEval 的判分逻辑
# ============================================================

def gen_acc_strict(x: Dict[str, Any]) -> Dict[str, List[bool]]:
    response = str(x["response"])
    instruction_list = x["instruction_id_list"]

    is_following_list = []
    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = ifeval.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**x["kwargs"][index])

        if response and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return {
        "follow_instruction_list": is_following_list,
        "instruction_id_list": instruction_list,
    }


def gen_acc_loose(x: Dict[str, Any]) -> Dict[str, List[bool]]:
    response = str(x["response"])

    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()

    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")

    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    instruction_list = x["instruction_id_list"]
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = ifeval.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**x["kwargs"][index])

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return {
        "follow_instruction_list": is_following_list,
        "instruction_id_list": instruction_list,
    }


# ============================================================
# JSON 友好的封装
# ============================================================

def eval_one_turn(
    response: str,
    instruction_id_list: List[str],
    kwargs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate one turn response against its instructions
    """
    input_dict = {
        "response": response,
        "instruction_id_list": instruction_id_list,
        "kwargs": kwargs,
    }

    strict_res = gen_acc_strict(input_dict)
    loose_res = gen_acc_loose(input_dict)

    return {
        "instruction_id_list": instruction_id_list,
        "follow_strict": strict_res["follow_instruction_list"],
        "follow_loose": loose_res["follow_instruction_list"],
    }


def eval_multi_if_sample(
    sample: Dict[str, Any],
    generated_responses: List[str],
) -> Dict[str, Any]:
    """
    Evaluate one Multi-IF sample (all turns)
    """
    result = {
        "key": sample.get("key"),
        "language": sample.get("language"),
        "turns": [],
    }

    for turn_id, response in enumerate(generated_responses, start=1):
        # instruction ids
        instruction_id_list = json.loads(
            sample[f"turn_{turn_id}_instruction_id_list"]
        )

        # kwargs
        kwargs_raw = json.loads(sample[f"turn_{turn_id}_kwargs"])
        kwargs = [json.loads(kw) for kw in kwargs_raw]

        # prompt
        prompt = json.loads(sample[f"turn_{turn_id}_prompt"])["content"]

        eval_res = eval_one_turn(
            response=response,
            instruction_id_list=instruction_id_list,
            kwargs=kwargs,
        )

        result["turns"].append(
            {
                "turn_id": turn_id,
                "prompt": prompt,
                "response": response,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
                "follow_strict": eval_res["follow_strict"],
                "follow_loose": eval_res["follow_loose"],
            }
        )

    return result
