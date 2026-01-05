# flake8: noqa: E501
import json
import os
import os.path as osp
import re
import sys
import time
import urllib.request
import urllib.error
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from opencompass.utils import get_data_path
except Exception:  # pragma: no cover - optional dependency
    def get_data_path(path: str, local_mode: bool = True) -> str:
        return path

try:
    from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
except Exception:  # pragma: no cover - optional dependency
    class _Registry:
        def register_module(self, name: Optional[str] = None):
            def decorator(obj):
                return obj
            return decorator

    DICT_POSTPROCESSORS = _Registry()
    LOAD_DATASET = _Registry()

try:
    from ..base import BaseDataset
except Exception:  # pragma: no cover - optional dependency
    class BaseDataset:  # type: ignore
        pass

try:
    from .utils import get_judgeanswer_and_reference
except Exception:  # pragma: no cover - optional dependency
    def get_judgeanswer_and_reference(output, output_path, post_process):
        if isinstance(output, dict):
            outputs = output.get("results") or []
        else:
            outputs = output or []

        judged_answers = []
        references = []
        for item in outputs:
            prediction = item.get("judge_prediction") or item.get("prediction") or ""
            processed = post_process({"prediction": prediction}) if post_process else None
            if processed is None:
                continue
            judged_answers.append(processed)
            references.append(
                item.get("judge")
                or item.get("reference")
                or {"task": item.get("task"), "multi_id": item.get("multi_id")}
            )
        return judged_answers, references
skip_first_tasks = ['FR', 'CR', 'AR', 'SA', 'SC', 'CM']

need_ref_tasks = ['MR', 'GR']

judge = "Please act as an impartial judge follow this instructions: In the following conversations, only the response of the 'assistant' in the last round of conversations is the output of the large language model (AI assistant) that needs to be evaluated.  Please act as an impartial judge and score this response on a scale of 1 to 10, where 1 indicates that the response completely fails to meet the criteria, and 10 indicates that the response perfectly meets all the evaluation criteria.\
    Note that only the response of the 'assistant' in the LAST ROUND of conversations is the output of the large language model (the AI assistant) that needs to be evaluated; the previous conversations is the groud truth history which do NOT need to be evaluated."

score_format = "\n\n Note that only the response of the 'assistant' in the LAST ROUND of conversations is the output of the large language model (the AI assistant) that needs to be evaluated!! You must provide your explanation. After providing your explanation, please show the score by strictly following this format: 'Rating: [[score]]', for example 'Rating: [[6]]'. The DIALGUE need to be judged is in this format: \n *** \n DIALGUE \n ***"

eval_CM = "The capacity of a large language model to recall and utilize previously mentioned information from earlier in the conversation is a critical indicator of its conversational memory abilities. This competency is essential for maintaining context and coherence throughout an extended dialogue. The performance of the AI assistant should be evaluated based on its ability to consistently reference and integrate past information into current responses. The evaluation criteria are as follows:\n\
\n\
1.Analyze whether the AI assistant appropriately recalls relevant details from earlier parts of the conversation when responding to 'Human's inquiries or comments.\n\
2.Assess the AI assistant's ability to integrate the remembered information into its current responses in a way that is coherent and adds value to the dialogue.\n\
3.Examine the AI assistant's consistency in maintaining the context established by previous dialogue exchanges throughout the entire conversation.\n\
4.Evaluate the effectiveness of the AI assistant's memory recall in facilitating a smooth and logical progression of the conversation, avoiding repetitive or contradictory statements.\n\
Scoring Guidelines:\n\
\n\
1-3 points: The AI assistant demonstrates poor recall of previous conversation details, leading to inconsistent or contradictory responses, and fails to maintain the dialogue's context, resulting in a disjointed or unclear conversation flow.\n\
4-6 points: The AI assistant exhibits a moderate ability to remember past information, but its integration into the conversation is sporadic or partially effective, leading to a conversation that lacks full coherence or occasionally disregards established context.\n\
7-9 points: The AI assistant reliably recalls and utilizes earlier information, contributing to a coherent dialogue that respects the conversation's context, with minor lapses in memory that do not significantly disrupt the conversation flow.\n\
10 points: The AI assistant demonstrates exceptional memory recall, seamlessly weaving past details into current responses to enrich the dialogue and preserve context, ensuring a smooth and logical conversation that progresses naturally.\n\
When scoring, consider the significance of the AI assistant's memory recall to the overall quality of the conversation. If recalling past information was not necessary for a particular exchange, the AI assistant's failure to reference earlier dialogue should not impact the score negatively. However, if recalling previous information enhances the dialogue's clarity, relevance, and continuity, this should be regarded as a positive attribute of the language model's performance.\n\
\n\
Please provide a rationale for your score, specifically addressing how the AI assistant's memory recall and the use of past information align with the evaluation criteria and contribute to the conversation's effectiveness."

eval_SI = "\n We aim to specifically evaluate the command-following ability of the large language model (AI assistant). The criteria for evaluation are as follows:\
\n \
1. In the first round, 'Human' will present a task request without providing details about what needs to be done. If the AI Assistant being evaluated generates a response for the first round, it should ask 'Human' for the specific details of the task required or wait for 'Human' to provide specific details of the required tasks, rather than directly attempting to answer the task.\
2. Starting from the second round, 'Human' will provide the specific content of what needs to be carried out for the task, without repeating the task requirement. The AI Assistant being evaluated should then provide correct and specific answers directly addressing the task requirements.\
\n \
Please rate the AI assistant's response using a 1 to 10 scale based on the following guidelines:\
\n \
- 1-3 points: The AI assistant failed to understand the ta///sk request and neither asked relevant questions nor provided information related to the task.\
- 4-6 points: The AI assistant understood some aspects of the task request but the response could be more specific or relevant.\
- 7-9 points: The AI assistant provided a useful response that was mostly correct and targeted, even though there may be minor oversights.\
- 10 points: The AI assistant demonstrated a perfect understanding of the task requirements and provided a comprehensive and accurate answer, fully meeting 'Human's expectations.\
\n \
Additionally, please provide a brief justification for the score given, particularly highlighting how the AI assistant's response aligns with or deviates from the above criteria. This will help us understand the performance of the AI assistant and take steps for improvement if necessary."

eval_CR = "\nWe aim to specifically evaluate the paraphrasing ability of the large language model (AI assistant). The criteria for evaluation are as follows:\n \
\n \
1. The content of the AI assistant's rewritten response must maintain the same main idea as the Assistant's response in the first round.\n \
2. The rewritten content must comply with the specific rewriting requirements set forth by the Human in the current round.\n \
\n \
Scoring Guidelines:\n \
\n \
- 1-3 points: The rewritten response significantly deviates from the original main idea or fails to meet the rewriting requirements.\n \
- 4-6 points: The rewritten response captures the original main idea but only partially meets the rewriting requirements or lacks fluency/coherence.\n \
- 7-9 points: The rewritten response maintains the original main idea and satisfies most of the rewriting requirements with minor discrepancies or stylistic issues.\n \
- 10 points: The rewritten response perfectly preserves the original main idea and fulfills all of the rewriting requirements set by Human, exhibiting a seamless and natural integration of the required changes.\n \
\n \
Please provide a brief justification for the score you give and present your score. Please judge the response and Do Not answer the question in the dialogue directly."

eval_FR = "\nWe aim to specifically evaluate the paraphrasing ability of the large language model (AI assistant). The criteria for evaluation are as follows:\n \
\n \
1. The content of the AI assistant's rewritten response must maintain the same main idea as the Assistant's response in the first round.\n \
2. The rewritten content must comply with the specific rewriting requirements set forth by the Human in the current round.\n \
\n \
Scoring Guidelines:\n \
\n \
- 1-3 points: The rewritten response significantly deviates from the original main idea or fails to meet the rewriting requirements.\n \
- 4-6 points: The rewritten response captures the original main idea but only partially meets the rewriting requirements or lacks fluency/coherence.\n \
- 7-9 points: The rewritten response maintains the original main idea and satisfies most of the rewriting requirements with minor discrepancies or stylistic issues.\n \
- 10 points: The rewritten response perfectly preserves the original main idea and fulfills all of the rewriting requirements set by Human, exhibiting a seamless and natural integration of the required changes.\n \
\n \
Please provide a brief justification for the score you give and present your score. Please judge the response and Do Not answer the question in the dialogue directly."

eval_CC = "\nThe AI assistant's capability to resist interference will be measured against these criteria:\
\n\n \
1. The AI assistant's response must directly correspond to the content of the Human's question in the current round, providing true and accurate information.\n\
2. The response must not be influenced by the question and answer pattern from the previous dialogue, ensuring that it remains relevant and focused on the current question only.\
\n\n \
Scoring Guidelines:\
\n\n \
- 1-3 points: The AI assistant's response is largely influenced by previous interactions, fails to address the current question accurately, or provides false information.\n\
- 4-6 points: The AI assistant's response shows some resistance to interference but includes irrelevant details from previous dialogues or only partially addresses the current question.\n\
- 7-9 points: The AI assistant's response is mostly resistant to interference and accurately addresses the current question, with only minor relevancies to previous interactions.\n\
- 10 points: The AI assistant's response is completely free from interference, focusing solely on the current question and providing a response that is both accurate and wholly relevant.\
\n\n \
Please provide a brief justification for the score you give, focusing on how well the AI assistant's response aligns with the two evaluation criteria. "

eval_TS = "\nThe AI assistant's ability to handle shifts in conversation topics is crucial for maintaining relevance and adaptability during a dialogue. This skill is particularly important when 'Human' introduces a new topic or changes the subject abruptly. The performance of the AI assistant should be evaluated on its capacity to smoothly transition between topics without being inappropriately influenced by previous dialogue content. The evaluation criteria are as follows:\n \
\n \
1. Identify whether the AI assistant can detect and acknowledge the change in topic introduced by 'Human' without reverting back to or becoming stuck on the previous subject.\n \
2. Evaluate the relevance of the AI assistant's responses to the new topic, ensuring they are not improperly influenced or colored by the preceding dialogue rounds.\n \
3. Assess the AI assistant's ability to provide coherent and contextually appropriate responses to the new subject, displaying an understanding of the conversation's evolving nature.\n \
4. Consider the AI assistant's proficiency in offering complete and insightful answers to the new topic, which demonstrate a clear break from past conversation threads.\n \
Scoring Guidelines:\n \
\n \
1-3 points: The AI assistant struggles with topic transitions, frequently reverting to or being influenced by the previous topic, resulting in irrelevant or confused responses to the new subject matter.\n \
4-6 points: The AI assistant shows a moderate ability to adapt to new topics, but occasionally exhibits lingering effects from earlier discussions, leading to partially relevant or less focused responses to the topic shifts.\n \
7-9 points: The AI assistant adapts to topic changes well, with minimal reference to or influence from prior topics, providing responses that are largely relevant and well-aligned with the new conversation direction.\n \
10 points: The AI assistant excels at adapting to topic shifts, seamlessly transitioning to and fully engaging with the new subject matter without any irrelevant carryover from previous dialogue content.\n \
When scoring, consider the smoothness of the AI assistant's transition between topics and its ability to engage with the new subject matter independently of the prior conversation. If a topic shift is not present or is so subtle that continuity with previous content is warranted, the AI assistant's ability to maintain coherence should not negatively affect the score. However, if a clear topic shift occurs and the AI assistant handles it deftly, providing relevant and insightful input on the new topic, this should be recognized as a positive aspect of its conversational capabilities.\n \
\n \
Please provide a rationale for your score, specifically addressing the effectiveness of the AI assistant's topic transition and its relevance to the new subject matter in accordance with the evaluation criteria."

eval_AR = "The AI assistant's understanding of references is essential for maintaining a coherent dialogue. The following criteria should be used to evaluate its performance:\n\
\n \
1. The AI assistant's response must demonstrate a correct understanding of referential information from questions asked by 'Human,' which typically relate to content from the previous dialogue. Ideally, the AI should explicitly acknowledge or clarify these references in its reply.\n\
2. The response from the AI assistant should be consistent with the content of the 'Human's question in the current round, providing true and accurate information, free from misunderstandings or inaccuracies related to the references.\n\
\n \
Scoring Guidelines:\n\
\n\
- 1-3 points: The AI assistant fails to recognize or correctly interpret the referential information, leading to responses that are either inaccurate or unrelated to the previous content.\n\
- 4-6 points: The AI assistant shows a partial understanding of references, but the response might include some inaccuracies or fail to fully utilize the referential information.\n\
- 7-9 points: The AI assistant's response indicates a good understanding of the references, with only slight inaccuracies or omissions in the connection to the previous dialogue.\n\
- 10 points: The AI assistant demonstrates excellent understanding and use of referential information, perfectly aligning its response with the previous content and the current question accurately and precisely.\n\
\n \
In addition to the score, please provide an explanation that specifically addresses how the AI assistant's response demonstrates its ability or inability to understand and use referential information in accordance with the criteria above. "

eval_IC = "The AI assistant’s ability to engage in a productive dialogue is often enhanced by its use of counter-questions, particularly when dealing with incomplete or vague queries. The assistant's performance should be assessed based on its ability to recognize when a rhetorical question is necessary and to use it effectively to clarify the 'Human's intent. The evaluation criteria are as follows:\n \
\n \
1. Assess whether the question posed by 'Human' contains ambiguities or lacks specific details that would require the AI assistant to use a counter-questions for clarification.\n \
2. If the question does require clarification through a counter-question, evaluate how the AI assistant employs this strategy to address the ambiguities or missing information in 'Human's query.\n \
3. Once 'Human' provides the necessary conditions or clarifies the question, evaluate whether the AI assistant offers a true and detailed response that fully addresses the clarified query.\n \
\n \
Scoring Guidelines:\n \
\n \
- 1-3 points: The AI assistant fails to identify the need for a rhetorical question when necessary, or it employs rhetorical questions ineffectively, leading to answers that do not align with 'Human's query, or lack the detail required to fully clarify the question.\n \
- 4-6 points: The AI assistant recognizes situations requiring rhetorical questions but uses them suboptimally, only partially addressing the query's deficiencies. Subsequent answers may lack full detail or accuracy even after the query is clarified.\n \
- 7-9 points: The AI assistant effectively uses rhetorical questions to pinpoint and address the missing or unclear elements in 'Human's query, and provides a largely accurate and detailed response to the perfected question.\n \
- 10 points: The AI assistant expertly discerns when to use rhetorical questions and employs them precisely to address the ambiguities or missing information in the query. Once clarified, it responds with detailed, accurate information that perfectly satisfies the question.\n \
\n \
When scoring, consider whether the use of a counter-question was essential and whether the AI assistant's decision to use or not use one improved the clarity and outcome of the dialogue. If a counter-question was not necessary, and the AI assistant refrained from using one, this should not negatively affect the score. However, if the use of a rhetorical question or follow-up query by the AI assistant brought clarity to an otherwise ambiguous situation, this should be seen as a positive contribution to the dialogue.\n \
\n \
Please provide a rationale for your score, specifically addressing how the AI assistant's use or omission of rhetorical questions and its responses align with the evaluation criteria and the necessity of such an approach for each particular query."

eval_SC = "We aim to specifically evaluate the self-correction ability of the large language model (AI assistant) when the 'Human' identifies an error in the AI assistant's initial response. The criteria for evaluation are as follows:\n\
\n\
1. Upon 'Human' pointing out a potential mistake, the AI assistant must thoroughly assess its previous response and engage with 'Human's' observation.\n\
2. If the 'Human' has correctly identified an error, the AI assistant must acknowledge the mistake, correct it, and provide an updated and accurate response.\n\
\n\
Scoring Guidelines:\n\
\n\
- 1-3 points: The AI assistant fails to recognize or adequately address the error identified by 'Human,' and does not make the necessary corrections to its response.\n\
- 4-6 points: The AI assistant recognizes the error identified by 'Human' but may only partially correct the mistake or provide an incomplete updated response.\n\
- 7-9 points: The AI assistant correctly identifies and acknowledges the error, making a substantial correction to its response and effectively updating the answer.\n\
- 10 points: The AI assistant exhibits exceptional self-correction ability, promptly acknowledging the error and providing a comprehensive and precise updated response.\n\
\n\
In addition to scoring, please provide a justification for your assessment, focusing on how effectively the AI assistant's reaction to 'Human's' identified error demonstrates its ability to self-correct and address the criteria outlined above."

eval_SA = "We aim to specifically evaluate the ability of the large language model (AI assistant) to maintain confidence in its original response when challenged by 'Human', provided that the challenge is incorrect. The criteria for evaluation are as follows:\n\
\n\
1. When 'Human' questions the content of the AI assistant's response, the AI assistant must engage with 'Human's' query and assess the validity of the challenge.\n\
2. If 'Human's challenge is based on incorrect information or assumptions, the AI assistant must confidently maintain its original answer and provide a clear and reasoned explanation to justify its stance.\n\
\n\
Scoring Guidelines:\n\
\n\
- 1-3 points: The AI assistant either does not engage with 'Human's' challenge appropriately or changes its original response without proper justification, even when 'Human's' challenge is incorrect.\n\
- 4-6 points: The AI assistant engages with the challenge but may lack confidence in its response, providing a weak justification for maintaining its original answer.\n\
- 7-9 points: The AI assistant demonstrates the ability to appropriately assess the incorrect challenge and maintains its original response with a clear and well-supported justification.\n\
- 10 points: The AI assistant exhibits excellent ability to maintain confidence in its original response, providing a strong and convincing explanation that effectively addresses 'Human's' incorrect challenge.\n\
\n\
In addition to scoring, please provide a justification for your assessment, focusing on how the AI assistant's reaction to the challenge reflects its understanding and confidence in its original response, and how well it meets the criteria outlined above."

eval_PI = "The AI assistant's interactivity, represented by its ability to proactively initiate and sustain engaging dialogues with 'Human', is a key aspect of a dynamic conversational experience. The model should not only respond passively but should also contribute to the momentum of the conversation by introducing questions, suggesting topics, or encouraging further discourse. The performance of the AI assistant should be evaluated on its capacity for active engagement and conversational leadership. The evaluation criteria are as follows:\n\
\n\
1. Observe the AI assistant's initiative in contributing to the conversation beyond providing direct answers, including its ability to ask relevant follow-up questions or propose new topics.\n\
2. Assess the AI assistant's aptness in maintaining the flow of the conversation, including how well it encourages 'Human' to provide more information or share their thoughts.\n\
3. Examine the appropriateness of the AI assistant's interactive elements in the context of the dialogue, ensuring they foster a natural and engaging conversation rather than derailing it.\n\
4. Evaluate the AI assistant's responsiveness to 'Human's input while being proactive, ensuring that it listens and adapts to the conversation's direction as set by 'Human'.\n\
Scoring Guidelines:\n\
\n\
1-3 points: The AI assistant exhibits poor interactivity, often providing minimal responses without encouraging further dialogue, or its attempts at interactivity are misplaced and hamper the natural flow of conversation.\n\
4-6 points: The AI assistant demonstrates moderate interactivity; it occasionally asks questions or suggests new topics but may not consistently maintain the conversational momentum or fully engage 'Human'.\n\
7-9 points: The AI assistant is highly interactive, regularly using questions and topics to keep the conversation going, while mostly preserving relevancy and a natural exchange with 'Human'.\n\
10 points: The AI assistant excels at interactivity, skillfully using questions and dialogue prompts to enrich the conversation, actively engaging 'Human', and enhancing the overall dialogue experience without dominating the conversation.\n\
When scoring, consider the balance the AI assistant strikes between guiding the conversation and allowing 'Human' to steer the dialogue. The AI assistant's interactivity should feel like a natural extension of the conversation, not forced or distracting from 'Human's intent. If the conversation benefits from the AI assistant's interactive elements, leading to a richer dialogue, this should be reflected in a higher score.\n\
\n\
Please provide a rationale for your score, specifically addressing how the AI assistant's proactive contributions and interactive strategies align with the evaluation criteria and enrich the conversational experience."

eval_MR = "The AI assistant's mathematical reasoning capabilities are vital for accurately solving and explaining mathematical problems posed by 'Human'. The model should leverage both the conditions provided in the current question and any relevant information from the historical dialogue. The evaluation of the AI assistant's performance will be based on the correctness of its answers and the clarity of its reasoning process. The evaluation criteria are as follows:\n\
\n\
1. Verify the accuracy of the AI assistant's answer against the provided reference solution in the format '### reference solution ###'  for the mathematical problem.\n\
2. Assess the completeness and step-by-step clarity of the AI assistant's reasoning process, ensuring it is logical and follows mathematical principles.\n\
3. Evaluate the AI assistant's ability to incorporate any relevant historical dialogue information that influences the problem-solving process or the solution itself.\n\
4. Appraise the AI assistant's communication of the solution in a manner that is understandable and instructive to 'Human', potentially aiding their learning or comprehension.\n\
Scoring Guidelines:\n\
\n\
1-3 points: The AI assistant provides incorrect answers and/or fails to offer a clear and logical reasoning process, missing key steps or providing explanations that do not align with mathematical standards.\n\
4-6 points: The AI assistant's answer is partially correct with minor errors in the reasoning process, which may lack detail or clarity in some steps, but generally follows mathematical principles.\n\
7-9 points: The AI assistant gives correct answers with a reasoning process that includes most necessary steps and details, facilitating a good understanding of the solution.\n\
10 points: The AI assistant provides a completely correct answer accompanied by a detailed and meticulously clear step-by-step reasoning process that is fully aligned with mathematical principles and enhances 'Human's understanding.\n\
When scoring, focus on the precision of the AI assistant's answer and the extent to which the reasoning process is elaborated. The assistant's ability to effectively communicate complex mathematical solutions in a manner that supports 'Human's learning is indicative of high performance. If the reasoning process is exemplary and the answer is accurate, this should be reflected in a top score.\n\
\n\
Please provide a rationale for your score, specifically addressing the accuracy of the AI assistant's answer and the quality of the mathematical reasoning process, considering the evaluation criteria and the comparison with the reference solution."

eval_GR = "The AI assistant's general reasoning capabilities are crucial for accurately addressing and explaining a wide range of problems posed by 'Human'. The evaluation of the AI assistant's performance will be based on the correctness of its answers and the cogency of its reasoning process. The evaluation criteria are as follows:\n\
\n\
1. Verify the accuracy of the AI assistant's answer against the provided reference solution in format ‘### reference solution ###‘ for the specific problem.\n\
2. Assess the completeness and step-by-step clarity of the AI assistant's reasoning process, ensuring it is logical and follows the principles of sound reasoning.\n\
3. Evaluate the AI assistant's ability to integrate any relevant historical dialogue information that influences the problem-solving process or the solution itself.\n\
4. Appraise the AI assistant's communication of the solution in a manner that is understandable and instructive to 'Human', potentially aiding their learning or comprehension.\n\
Scoring Guidelines:\n\
\n\
1-3 points: The AI assistant provides incorrect answers and/or fails to offer a clear and logical reasoning process, missing key steps or providing explanations that do not adhere to standards of sound reasoning.\n\
4-6 points: The AI assistant's answer is partially correct with minor errors in the reasoning process, which may lack detail or clarity in some steps but generally follows sound reasoning principles.\n\
7-9 points: The AI assistant gives correct answers with a well-articulated reasoning process that includes most necessary steps and details, facilitating a good understanding of the solution.\n\
10 points: The AI assistant provides a completely correct answer accompanied by a detailed and meticulously clear step-by-step reasoning process that is fully aligned with sound reasoning principles and enhances 'Human's understanding.\n\
When scoring, focus on the precision of the AI assistant's answer and the extent to which the reasoning process is elaborated. The assistant's ability to effectively communicate complex solutions in a manner that supports 'Human's learning is indicative of high performance. If the reasoning process is exemplary and the answer is accurate, this should be reflected in a top score.\n\
\n\
Please provide a rationale for your score, specifically addressing the accuracy of the AI assistant's answer and the quality of the general reasoning process, considering the evaluation criteria and the comparison with the reference solution."

unique_prompt = {
    'CM': eval_CM,
    'SI': eval_SI,
    'AR': eval_AR,
    'TS': eval_TS,
    'CC': eval_CC,
    'CR': eval_CR,
    'FR': eval_FR,
    'SC': eval_SC,
    'SA': eval_SA,
    'MR': eval_MR,
    'GR': eval_GR,
    'IC': eval_IC,
    'PI': eval_PI,
}


def eval_prompt_construct(task, ref_answer, history):

    if task in need_ref_tasks:
        system_prompt = judge + unique_prompt[task] + score_format
        prompt_template = 'The dialogue need to be judged is: \n *** \n {history} {prediction} \n ***\n\n\
                    The reference solution is: \n ### \n {ref_answer} \n ###\n\n'.format(
            history=history, prediction='{prediction}', ref_answer=ref_answer)

    else:
        system_prompt = judge + unique_prompt[task] + score_format
        prompt_template = 'The dialogue need to be judged is: \n *** \n {history} {prediction} \n ***'.format(
            history=history, prediction='{prediction}')

    return system_prompt, prompt_template


def add_format(question, answer):
    history = [dict(role='user', content=question)]
    if answer:
        history += [dict(role='assistant', content=answer)]
    return history


@LOAD_DATASET.register_module()
class MTBench101Dataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        import copy
        from datasets import Dataset, DatasetDict

        filename = osp.join(path, f'{name}.jsonl')
        filename = get_data_path(filename, local_mode=True)
        # filename = osp.join(path, 'mtbench_101.jsonl')
        dataset = DatasetDict()
        raw_data = []

        lines = open(filename, 'r', encoding='utf-8').readlines()
        conversations = []
        for line in lines:
            line = json.loads(line)
            conversations.append(line)

        for dialogue in conversations:
            multi_id = dialogue['id']
            task = dialogue['task']
            if task in skip_first_tasks:
                skip_first = True
            else:
                skip_first = False

            current_multi_id = None
            pre_dia = []
            history = ''
            dia_list = []
            for turn_index, turn in enumerate(dialogue['history']):
                human = turn['user']
                assistant = turn['bot']
                turn_id = str(turn_index + 1)

                if current_multi_id is not None and multi_id != current_multi_id:
                    pre_dia = []
                    history = ''

                current_multi_id = multi_id

                if skip_first and turn_index == 0:
                    pre_dia = add_format(question=human, answer=assistant)
                    history = '\n\n Human: ' + human + '\n\nAssistant: ' + assistant
                    continue

                history = history + '\n\n Human: ' + human + '\n\nAssistant: '
                pre_dia += add_format(question=human, answer=assistant)

                pre_dia_copy = copy.deepcopy(pre_dia)

                system_prompt, prompt_template = eval_prompt_construct(
                    task, pre_dia, history)

                raw_data.append({
                    'dialogue': pre_dia_copy,
                    'task': task,
                    'multi_id': current_multi_id,
                    'turn_id': turn_id,
                    'system_prompt': system_prompt,
                    'prompt_template': prompt_template,
                    'judge': {
                        'task': task,
                        'multi_id': current_multi_id,
                        'turn_id': turn_id,
                    }
                })
                history = history + assistant

        dataset = Dataset.from_list(raw_data)
        return dataset


def post_process_mtbench101(judgement: str):
    """Input a string like below:

    xxx[[5]]xxx, and extract the score
    """
    judgement = judgement['prediction']
    match = re.search(r'\[([0-9]+)\]', judgement)
    if match:
        score = int(match.group(1))

    else:
        return None

    return {'score': score, 'judgement': judgement}


def get_final_results(judged_answers, references):

    task_multi_id_scores = defaultdict(list)
    task_scores = defaultdict(list)

    for ans, ref in zip(judged_answers, references):

        task = ref['task']
        multi_id = ref['multi_id']
        score = ans['score']

        task_multi_id_scores[(task, multi_id)].append(score)

    for (task, multi_id), scores in task_multi_id_scores.items():
        min_score = min(scores)
        task_scores[task].append(min_score)

    final_task_scores = {
        task: sum(scores) / len(scores) if scores else 0
        for task, scores in task_scores.items()
    }
    average_score = round(
        sum(final_task_scores.values()) / len(final_task_scores), 3)

    return {f'avg': average_score, **final_task_scores}


@DICT_POSTPROCESSORS.register_module('mtbench101')
def mtbench101_postprocess(output: dict,
                           output_path: str,
                           ) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_mtbench101)

    results = get_final_results(judged_answers, references)
    results['details'] = output
    return results


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _parse_cuda_ids(cuda_id: Any, cuda_ids: Optional[Any]) -> List[int]:
    source = cuda_ids if cuda_ids is not None else cuda_id
    if source is None:
        return [0]

    if isinstance(source, (list, tuple)):
        ids = [int(x) for x in source]
    elif isinstance(source, str):
        s = source.strip()
        if not s:
            ids = [0]
        else:
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        ids = [int(x) for x in parsed]
                    else:
                        ids = [0]
                except Exception:
                    ids = [int(x) for x in re.split(r"[,\s]+", s) if x]
            else:
                ids = [int(x) for x in re.split(r"[,\s]+", s) if x]
    else:
        ids = [int(source)]

    return list(dict.fromkeys(ids))


def _validate_cuda_ids(cuda_ids: List[int]) -> None:
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        if cuda_ids and cuda_ids != [0]:
            print(f"[WARN] CUDA not available; falling back to CPU (cuda_ids={cuda_ids}).")
        return

    device_count = torch.cuda.device_count()
    for cid in cuda_ids:
        if cid < 0 or cid >= device_count:
            raise ValueError(
                f"Invalid cuda_id={cid}; torch.cuda.device_count()={device_count}."
            )


def load_beacon_model(model_path: str, device_id: int = 0, num_sinks: int = 1):
    import torch
    from transformers import AutoConfig
    from modeling_qwen3 import Qwen3ForCausalLM as BeaconQwen3ForCausalLM

    print(f"Loading Beacon model: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_sinks = num_sinks

    print(f"num_sinks = {config.num_sinks}")
    print(f"num_beacons_per_segment = {config.num_beacons_per_segment}")

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    model = BeaconQwen3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    print(f"Beacon model loaded on {device}")
    return model, device


def encode_dialogue(dialogue: List[Dict[str, str]], tokenizer, device):
    text = tokenizer.apply_chat_template(
        dialogue,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    import torch
    input_ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def generate_response(
    model,
    input_ids,
    attention_mask,
    tokenizer,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: Optional[float] = None,
):
    import torch

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "enable_beacon_compression": True,
    }
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    input_len = input_ids.shape[1]
    new_tokens = output_ids[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text


def _normalize_base_url(base_url: str) -> str:
    return base_url.strip()


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120,
        max_retries: int = 3,
        retry_sleep: float = 1.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._client = None
        self._use_sdk = "/chat/completions" not in self.base_url
        if self._use_sdk:
            try:
                from openai import OpenAI
                try:
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                    )
                except TypeError:
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                    )
            except Exception:
                self._client = None

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
            except TypeError:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
            return response.choices[0].message.content

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(self.base_url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                choices = data.get("choices") or []
                if not choices:
                    raise ValueError(f"No choices in response: {data}")
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if content is None:
                    raise ValueError(f"Missing message content: {data}")
                return content
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep)
                    continue
                raise
        raise last_err  # pragma: no cover


def _load_api_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing config file: {config_path}. "
            "Create it with openai_api_key, openai_base_url, judge_model."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not cfg.get("openai_api_key"):
        raise ValueError("openai_api_key missing in config.")
    if not cfg.get("openai_base_url"):
        raise ValueError("openai_base_url missing in config.")
    if not cfg.get("judge_model"):
        raise ValueError("judge_model missing in config.")
    return cfg


def _load_mtbench101(path: str, name: Optional[str] = None) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl") and os.path.isfile(path):
        filename = path
    else:
        if not name:
            raise ValueError("name is required when path is a directory.")
        filename = osp.join(path, f"{name}.jsonl")
    filename = get_data_path(filename, local_mode=True)

    conversations = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))
    return conversations


def _get_turn_user(turn: Dict[str, Any]) -> str:
    return str(turn.get("user") or turn.get("human") or turn.get("prompt") or "")


def _get_turn_bot(turn: Dict[str, Any]) -> str:
    return str(turn.get("bot") or turn.get("assistant") or turn.get("response") or "")


def _build_history_str(
    turns: List[Dict[str, Any]],
    current_turn_index: int,
) -> str:
    history = ""
    for i in range(current_turn_index + 1):
        user = _get_turn_user(turns[i])
        history += f"\n\n Human: {user}\n\nAssistant: "
        if i < current_turn_index:
            history += _get_turn_bot(turns[i])
    return history


def _extract_reference(dialogue: Dict[str, Any], turn: Dict[str, Any], turn_index: int) -> str:
    ref_keys = [
        "reference",
        "reference_answer",
        "ref_answer",
        "reference_solution",
        "answer",
        "gold",
        "gold_answer",
    ]
    for key in ref_keys:
        val = turn.get(key)
        if val:
            return str(val)

    dialogue_ref = dialogue.get("reference") or dialogue.get("references")
    if isinstance(dialogue_ref, dict):
        for k in (str(turn_index + 1), str(turn_index), turn_index + 1, turn_index):
            if k in dialogue_ref and dialogue_ref[k]:
                return str(dialogue_ref[k])
    elif dialogue_ref:
        return str(dialogue_ref)

    bot = _get_turn_bot(turn)
    if bot:
        return bot
    return ""


def _evaluate_dialogue(
    dialogue: Dict[str, Any],
    dialogue_index: Optional[int],
    model,
    tokenizer,
    device,
    judge_client: OpenAICompatClient,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
) -> List[Dict[str, Any]]:
    task = dialogue.get("task", "")
    multi_id = dialogue.get("id", None)
    turns = dialogue.get("history") or []
    skip_first = task in skip_first_tasks
    results: List[Dict[str, Any]] = []

    for turn_index, turn in enumerate(turns):
        if skip_first and turn_index == 0:
            continue

        dialogue_messages: List[Dict[str, str]] = []
        system_prompt = dialogue.get("system") or dialogue.get("system_prompt")
        if system_prompt:
            dialogue_messages.append({"role": "system", "content": str(system_prompt)})
        for i in range(turn_index + 1):
            user = _get_turn_user(turns[i])
            dialogue_messages.append({"role": "user", "content": user})
            if i < turn_index:
                bot = _get_turn_bot(turns[i])
                if bot:
                    dialogue_messages.append({"role": "assistant", "content": bot})

        input_ids, attention_mask = encode_dialogue(dialogue_messages, tokenizer, device)
        pred = generate_response(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
        )

        history_str = _build_history_str(turns, turn_index)
        ref_answer = _extract_reference(dialogue, turn, turn_index) if task in need_ref_tasks else ""
        system_prompt, prompt_template = eval_prompt_construct(task, ref_answer, history_str)
        prompt = prompt_template.format(prediction=pred)

        judge_text = judge_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        judged = post_process_mtbench101({"prediction": judge_text})
        score = judged["score"] if judged else None

        results.append(
            {
                "task": task,
                "multi_id": multi_id,
                "turn_id": str(turn_index + 1),
                "dialogue_index": dialogue_index,
                "prediction": pred,
                "judge_prediction": judge_text,
                "score": score,
                "judge": {
                    "task": task,
                    "multi_id": multi_id,
                    "turn_id": str(turn_index + 1),
                },
            }
        )

    return results


def _worker_output_path(log_dir: str, timestamp: str, run_tag: str, worker_id: int) -> str:
    return os.path.join(
        log_dir, f"mtbench_101_beacon_{timestamp}_{run_tag}.worker{worker_id}.json"
    )


def _run_worker(
    *,
    worker_id: int,
    num_workers: int,
    cuda_id: int,
    model_path: str,
    num_sinks: int,
    log_dir: str,
    timestamp: str,
    run_tag: str,
    api_config: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
    flush_every: int,
    progress_counter: Optional[Any],
    task_queue: Any,
) -> None:
    from transformers import AutoTokenizer

    worker_out = _worker_output_path(log_dir, timestamp, run_tag, worker_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    model, device = load_beacon_model(
        model_path,
        device_id=cuda_id,
        num_sinks=num_sinks,
    )

    judge_client = OpenAICompatClient(
        base_url=api_config["openai_base_url"],
        api_key=api_config["openai_api_key"],
        model=api_config["judge_model"],
        timeout=api_config.get("request_timeout", 120),
        max_retries=api_config.get("max_retries", 3),
        retry_sleep=api_config.get("retry_sleep", 1.0),
        temperature=api_config.get("temperature", 0.0),
        top_p=api_config.get("top_p", 1.0),
        max_tokens=api_config.get("max_tokens", 512),
    )

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    processed_dialogues = 0

    def flush() -> None:
        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_id": cuda_id,
                "worker_id": worker_id,
                "num_workers": num_workers,
                "run_tag": run_tag,
                "num_sinks": num_sinks,
                "judge_model": api_config["judge_model"],
                "processed_dialogues": processed_dialogues,
                "attempted_items": len(results),
                "failed_dialogues": len(errors),
            },
            "results": results,
            "errors": errors,
        }
        try:
            write_json_atomic(worker_out, output)
        except Exception as e:
            print(f"[WARN] Failed to write {worker_out}: {e}")

    flush()

    processed = 0
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            
            idx, dialogue = task
            
            try:
                dialog_results = _evaluate_dialogue(
                    dialogue=dialogue,
                    dialogue_index=idx,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    judge_client=judge_client,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                )
                results.extend(dialog_results)
            except Exception as e:
                errors.append(
                    {
                        "dialogue_index": idx,
                        "id": dialogue.get("id"),
                        "task": dialogue.get("task"),
                        "error": repr(e),
                    }
                )
            finally:
                processed += 1
                processed_dialogues += 1
                if progress_counter is not None:
                    try:
                        with progress_counter.get_lock():
                            progress_counter.value += 1
                    except Exception:
                        pass
                if flush_every > 0 and (processed % flush_every == 0):
                    flush()
    finally:
        flush()


def main(
    model_path: str,
    data_path: str,
    name: Optional[str] = None,
    config_path: Optional[str] = None,
    cuda_id: int = 0,
    cuda_ids: Optional[str] = None,
    log_dir: Optional[str] = None,
    num_sinks: int = 1,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: Optional[float] = None,
    flush_every: int = 1,
    num_workers: int = 16,
) -> str:
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    api_config_path = config_path or os.path.join(
        PROJECT_ROOT, "eval", "mtbench_101", "mtbench_101_config.json"
    )
    api_config = _load_api_config(api_config_path)

    num_beacons_per_segment: Optional[int] = None
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_beacons_per_segment = getattr(cfg, "num_beacons_per_segment", None)
    except Exception as e:
        print(f"[WARN] Failed to read model config from {model_path}: {e}")

    beacon_part = (
        f"{int(num_beacons_per_segment)}beacon"
        if num_beacons_per_segment is not None
        else "unknownbeacon"
    )
    model_tag = osp.basename(model_path.rstrip("/"))
    run_tag = f"{model_tag}_{beacon_part}_{int(num_sinks)}sink"
    output_path = os.path.join(
        resolved_log_dir, f"mtbench_101_beacon_{timestamp}_{run_tag}.json"
    )

    conversations = _load_mtbench101(data_path, name=name)

    cuda_id_list = _parse_cuda_ids(cuda_id=cuda_id, cuda_ids=cuda_ids)
    _validate_cuda_ids(cuda_id_list)

    # Determine worker count
    final_num_workers = num_workers if (torch.cuda.is_available() and len(cuda_id_list) > 0) else 1
    if not torch.cuda.is_available():
        final_num_workers = 1

    print(f"Plan: {final_num_workers} workers on devices: {cuda_id_list or 'CPU'}")

    # Prepare Queue
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    
    for idx, dialogue in enumerate(conversations):
        task_queue.put((idx, dialogue))
        
    for _ in range(final_num_workers):
        task_queue.put(None)

    worker_paths = [
        _worker_output_path(resolved_log_dir, timestamp, run_tag, wid)
        for wid in range(final_num_workers)
    ]

    progress_counter = ctx.Value("i", 0)
    procs: List[mp.Process] = []
    
    for wid in range(final_num_workers):
        # Round-robin GPU assignment
        if cuda_id_list:
            cid = cuda_id_list[wid % len(cuda_id_list)]
        else:
            cid = 0

        p = ctx.Process(
            target=_run_worker,
            kwargs={
                "worker_id": wid,
                "num_workers": final_num_workers,
                "cuda_id": cid,
                "model_path": model_path,
                "num_sinks": num_sinks,
                "log_dir": resolved_log_dir,
                "timestamp": timestamp,
                "run_tag": run_tag,
                "api_config": api_config,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "flush_every": flush_every,
                "progress_counter": progress_counter,
                "task_queue": task_queue,
            },
        )
        p.start()
        procs.append(p)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(conversations), desc="Evaluating MTBench_101", unit="dialogue")
    
    last = 0
    try:
        while any(p.is_alive() for p in procs):
            try:
                with progress_counter.get_lock():
                    current = int(progress_counter.value)
            except Exception:
                current = last
            if current > last and pbar is not None:
                step = min(current - last, len(conversations) - last)
                if step > 0:
                    pbar.update(step)
                    last += step
            time.sleep(0.5)
            
        # Final update
        try:
            with progress_counter.get_lock():
                current = int(progress_counter.value)
        except Exception:
            current = last
        if current > last and pbar is not None:
            step = min(current - last, len(conversations) - last)
            if step > 0:
                pbar.update(step)
                last += step

    finally:
        if pbar is not None:
            pbar.close()

    for p in procs:
        p.join()

    failed_workers = []
    for wid, p in enumerate(procs):
        if p.exitcode != 0:
            failed_workers.append(
                {
                    "worker_id": wid,
                    "cuda_id": cuda_id_list[wid % len(cuda_id_list)] if cuda_id_list else 0,
                    "exitcode": p.exitcode,
                }
            )

    all_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for wp in worker_paths:
        if not os.path.exists(wp):
            print(f"[WARN] Missing worker output: {wp}")
            continue
        try:
            with open(wp, "r", encoding="utf-8") as f:
                d = json.load(f)
            all_results.extend(d.get("results") or [])
            errors.extend(d.get("errors") or [])
        except Exception as e:
            print(f"[WARN] Failed to read {wp}: {e}")

    all_results.sort(key=lambda r: (r.get("dialogue_index"), r.get("turn_id")))

    judged_answers = [{"score": r.get("score")} for r in all_results if r.get("score") is not None]
    references = [r.get("judge") for r in all_results if r.get("score") is not None]
    results = get_final_results(judged_answers, references) if judged_answers else {}

    output = {
        "meta": {
            "timestamp": timestamp,
            "model_path": model_path,
            "cuda_ids": cuda_id_list,
            "log_dir": resolved_log_dir,
            "num_workers": final_num_workers,
            "num_dialogues": len(conversations),
            "run_tag": run_tag,
            "num_beacons_per_segment": num_beacons_per_segment,
            "num_sinks": num_sinks,
            "judge_model": api_config["judge_model"],
            "attempted_items": len(all_results),
            "failed_dialogues": len(errors),
            "worker_outputs": worker_paths,
            "failed_workers": failed_workers,
        },
        "metrics": results,
        "results": all_results,
        "errors": errors,
    }
    write_json_atomic(output_path, output)

    print("\n=== MTBench_101 METRICS ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {output_path}")
    if failed_workers:
        raise SystemExit(
            f"{len(failed_workers)}/{final_num_workers} workers failed; "
            f"partial results saved to: {output_path}"
        )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MTBench_101 Beacon evaluation.")
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument(
        "--name",
        default=None,
        help="Dataset name when data-path is a directory.",
    )
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        default=None,
        help="Path to JSON config with openai_api_key/openai_base_url/judge_model.",
    )
    parser.add_argument("--cuda-id", "--cuda_id", dest="cuda_id", type=int, default=0)
    parser.add_argument("--cuda-ids", "--cuda_ids", dest="cuda_ids", default=None)
    parser.add_argument("--log-dir", "--log_dir", dest="log_dir", default=None)
    parser.add_argument("--num-sinks", "--num_sinks", dest="num_sinks", type=int, default=1)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", "--do_sample", dest="do_sample", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=None)
    parser.add_argument("--flush-every", "--flush_every", dest="flush_every", type=int, default=1)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=16)
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        data_path=args.data_path,
        name=args.name,
        config_path=args.config_path,
        cuda_id=args.cuda_id,
        cuda_ids=args.cuda_ids,
        log_dir=args.log_dir,
        num_sinks=args.num_sinks,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        flush_every=args.flush_every,
        num_workers=args.num_workers,
    )
