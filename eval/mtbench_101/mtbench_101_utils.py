import json
import os
import os.path as osp
import re
import time
import urllib.request
import urllib.error
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    from opencompass.utils import get_data_path
except Exception:  # pragma: no cover - optional dependency
    def get_data_path(path: str, local_mode: bool = True) -> str:
        return path

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

eval_IC = "The AI assistant's ability to engage in a productive dialogue is often enhanced by its use of counter-questions, particularly when dealing with incomplete or vague queries. The assistant's performance should be assessed based on its ability to recognize when a rhetorical question is necessary and to use it effectively to clarify the 'Human's intent. The evaluation criteria are as follows:\n \
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
1-3 points: The AI assistant fails to recognize or correct its mistake, either dismissing 'Human's' observation or providing an inaccurate response.\n\
4-6 points: The AI assistant acknowledges the error but provides only a partially correct response or lacks thorough engagement with the correction.\n\
7-9 points: The AI assistant corrects the mistake effectively and provides a mostly accurate response, though minor inaccuracies may remain.\n\
10 points: The AI assistant fully acknowledges the mistake, provides a thoroughly corrected and accurate response, and engages constructively with 'Human's' observation.\n\
\n\
Please provide a rationale for your score, specifically addressing how well the AI assistant responded to the correction and aligned with the evaluation criteria."

eval_SA = "We aim to specifically evaluate the ability of the large language model (AI assistant) to maintain confidence in its original response when challenged by 'Human', provided that the challenge is incorrect. The criteria for evaluation are as follows:\n\
\n\
1. When 'Human' questions the AI assistant's original response without a valid basis, the AI assistant should confidently reaffirm its original response if it is correct.\n\
2. The AI assistant should provide a clear rationale or explanation to justify its original response in the face of the incorrect challenge.\n\
\n\
Scoring Guidelines:\n\
\n\
1-3 points: The AI assistant fails to defend its correct response, either conceding to the incorrect challenge or providing a weak justification.\n\
4-6 points: The AI assistant somewhat defends its response but lacks clarity or confidence in its justification.\n\
7-9 points: The AI assistant confidently reaffirms its response with a reasonable justification, though minor gaps may exist.\n\
10 points: The AI assistant firmly and clearly defends its correct response, providing a strong and comprehensive justification.\n\
\n\
Please provide a rationale for your score, specifically addressing how well the AI assistant maintained confidence in its correct response and the quality of its justification."

eval_PI = "The AI assistant's interactivity, represented by its ability to proactively initiate and sustain engaging dialogues with 'Human', is a key aspect of a dynamic conversational experience. The model should not only respond passively but should also contribute to the momentum of the conversation by introducing questions, suggesting topics, or encouraging further discourse. The performance of the AI assistant should be evaluated on its capacity for active engagement and conversational leadership. The evaluation criteria are as follows:\n\
\n\
1. Assess whether the AI assistant demonstrates initiative in the conversation by asking relevant questions, suggesting topics, or inviting further discussion when appropriate.\n\
2. Evaluate the AI assistant's ability to sustain engagement by responding in a way that encourages the 'Human' to continue the dialogue.\n\
3. Consider the relevance and appropriateness of the AI assistant's proactive contributions to the flow and context of the conversation.\n\
4. Examine the AI assistant's balance between responding to 'Human's questions and introducing new directions for the conversation, ensuring it does not derail the discussion.\n\
Scoring Guidelines:\n\
\n\
1-3 points: The AI assistant is largely passive, providing minimal engagement and rarely initiating or sustaining meaningful conversation.\n\
4-6 points: The AI assistant shows some initiative but its contributions are inconsistent, only occasionally sustaining engagement.\n\
7-9 points: The AI assistant actively engages the 'Human' with relevant questions or suggestions, sustaining the conversation effectively with minor lapses.\n\
10 points: The AI assistant consistently drives engaging and proactive dialogue, striking an excellent balance between responding and initiating, enhancing the overall conversational experience.\n\
When scoring, consider whether the AI assistant's proactive behavior was appropriate and helped maintain or deepen the conversation. If the situation did not call for proactive engagement, the AI assistant should not be penalized for not taking initiative. Conversely, when proactive engagement could improve the dialogue, it should be rewarded.\n\
\n\
Please provide a rationale for your score, specifically addressing the AI assistant's level of interactivity and its effect on the conversation's engagement and flow."

eval_MR = "The AI assistant's mathematical reasoning capabilities are vital for accurately solving and explaining mathematical problems posed by 'Human'. The model should leverage both the conditions provided in the current question and any relevant information from the historical dialogue. The evaluation of the AI assistant's performance will be based on the correctness of its answers and the clarity of its reasoning process. The evaluation criteria are as follows:\n\
\n\
1. Verify the accuracy of the AI assistant's answer against the provided reference solution in format '### reference solution ###' for the specific problem.\n\
2. Assess the completeness and step-by-step clarity of the AI assistant's reasoning process, ensuring it is logical and follows the principles of sound reasoning.\n\
3. Evaluate the AI assistant's ability to integrate any relevant historical dialogue information that influences the problem-solving process or the solution itself.\n\
4. Appraise the AI assistant's communication of the solution in a manner that is understandable and instructive to 'Human', potentially aiding their learning or comprehension.\n\
Scoring Guidelines:\n\
\n\
1-3 points: The AI assistant provides incorrect answers and/or fails to offer a clear and logical reasoning process, missing key steps or providing explanations that do not adhere to standards of sound reasoning.\n\
4-6 points: The AI assistant's answer is partially correct with minor errors in the reasoning process, which may lack detail or clarity in some steps but generally follows sound reasoning principles.\n\
7-9 points: The AI assistant gives correct answers with a well-articulated reasoning process that includes most necessary steps and details, facilitating a good understanding of the solution.\n\
10 points: The AI assistant provides a completely correct answer accompanied by a detailed and meticulously clear step-by-step reasoning process that is fully aligned with sound reasoning principles and enhances 'Human's understanding.\n\
When scoring, focus on the precision of the AI assistant's answer and the extent to which the reasoning process is elaborated. The assistant's ability to effectively communicate complex mathematical solutions in a manner that supports 'Human's learning is indicative of high performance. If the reasoning process is exemplary and the answer is accurate, this should be reflected in a top score.\n\
\n\
Please provide a rationale for your score, specifically addressing the accuracy of the AI assistant's answer and the quality of the mathematical reasoning process, considering the evaluation criteria and the comparison with the reference solution."

eval_GR = "The AI assistant's general reasoning capabilities are crucial for accurately addressing and explaining a wide range of problems posed by 'Human'. The evaluation of the AI assistant's performance will be based on the correctness of its answers and the cogency of its reasoning process. The evaluation criteria are as follows:\n\
\n\
1. Verify the accuracy of the AI assistant's answer against the provided reference solution in format '### reference solution ###' for the specific problem.\n\
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


def post_process_mtbench101(judgement: str):
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


def mtbench101_postprocess(output: dict, output_path: str) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_mtbench101)

    results = get_final_results(judged_answers, references)
    results['details'] = output
    return results


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


def load_api_config(config_path: str) -> Dict[str, Any]:
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


def load_mtbench101(path: str, name: Optional[str] = None) -> List[Dict[str, Any]]:
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


def get_turn_user(turn: Dict[str, Any]) -> str:
    return str(turn.get("user") or turn.get("human") or turn.get("prompt") or "")


def get_turn_bot(turn: Dict[str, Any]) -> str:
    return str(turn.get("bot") or turn.get("assistant") or turn.get("response") or "")


def build_history_str(
    turns: List[Dict[str, Any]],
    current_turn_index: int,
) -> str:
    history = ""
    for i in range(current_turn_index + 1):
        user = get_turn_user(turns[i])
        history += f"\n\n Human: {user}\n\nAssistant: "
        if i < current_turn_index:
            history += get_turn_bot(turns[i])
    return history


def extract_reference(dialogue: Dict[str, Any], turn: Dict[str, Any], turn_index: int) -> str:
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

    bot = get_turn_bot(turn)
    if bot:
        return bot
    return ""
