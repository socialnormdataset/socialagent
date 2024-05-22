import os
import json
import time
import math
import torch
import argparse
import openai
import Levenshtein
import traceback
import pickle
from typing import Optional
from tqdm import tqdm
from openai import OpenAI
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat import ChatCompletion
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Conversation,
    ConversationalPipeline,
)
from agent_prompts import (
    WIKI_AGENT_PROMPTS,
    CALC_AGENT_PROMPTS,
    REASONING_AGENT_PROMPTS,
    USEFUL_AGENT_PROMPTS,
    CHOOSE_AGENT_PROMPTS,
    SUMM_AGENT_PROMPTS,
    WIKI_LLAMA_2_AGENT_PROMPTS,
    CALC_LLAMA_2_AGENT_PROMPTS,
    REASONING_LLAMA_2_AGENT_PROMPTS,
    USEFUL_LLAMA_2_AGENT_PROMPTS,
)
from loguru import logger
from collections import namedtuple
from langchain.utilities import WikipediaAPIWrapper

OPENAI_KEY = None
OPENAI_MODEL_NAME = None
WikiSearchRes = namedtuple("WikiSearchRes", "search_content search_res")
CalculatorRes = namedtuple("CalculatorRes", "raw_exp expression calc_res success")


class Llama2GenerationWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.generation_pipe = pipeline(
            task="conversational",
            model=self.llm,
            tokenizer=self.tokenizer,
            pad_token_id=self.llm.config.eos_token_id,
            do_sample=False,
        )


@dataclass
class LLMArgs:
    llm_max_rerun: int
    llm_wait_interval: float
    llm_client: OpenAI | Llama2GenerationWrapper
    is_local: bool


class MyOpenAIAgent:
    def __init__(
        self, agent_name: str, system_prompt: str, max_new_tokens: Optional[int]
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

    def run(self, input_str: str, openai_args: LLMArgs) -> ChatCompletion:
        system_prompt = {
            "role": "system",
            "content": self.system_prompt,
        }
        user_prompt = {
            "role": "user",
            "content": input_str,
        }
        prompts = [system_prompt, user_prompt]
        logger.debug(
            "\033[35mAgent\033[0m: {}\n\033[35mPrompts\033[0m: {}".format(
                self.agent_name, json.dumps(prompts, indent=2)
            )
        )
        curr_res = llm_call_wrapper(
            openai_args,
            messages=prompts,
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )
        logger.debug(
            "\033[35mAgent\033[0m: {}\n\033[35mOutput\033[0m: {}".format(
                self.agent_name, curr_res
            )
        )
        return curr_res


class CalculatorWrapper:
    def __init__(self):
        pass

    def run(self, input_json_str: str) -> tuple[str, str, bool]:
        success = True
        logger.debug(
            "\033[35mCalculator input_json_str\033[0m: {}".format(input_json_str)
        )
        try:
            input_str = json.loads(input_json_str)["Expression"]
            if "input()" in input_str:
                raise ValueError("input() in expression.")
            output = str(eval(input_str))
        except Exception as e:
            logger.debug("Calculator eval failed.")
            logger.debug(
                "\033[33mCalculator error:\033[0m {}".format(traceback.format_exc())
            )
            input_str = input_json_str
            output = ""
            success = False
        return input_str, output, success


class WikiSearcher:
    def __init__(self, cache_dir="wiki_search_cache.pkl"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            self.wiki_search_cache = {}
        else:
            with open(cache_dir, "rb") as f:
                self.wiki_search_cache = pickle.load(f)
        self.wiki_tool = WikipediaAPIWrapper()
        self.wiki_tool.top_k_results = 2
        self.wiki_tool.doc_content_chars_max = 1024
        self.max_rerun = 5

    def search(self, keyword: str) -> WikiSearchRes:
        if keyword in self.wiki_search_cache:
            logger.debug("Wiki search cache hit.")
            return WikiSearchRes(
                search_content=keyword, search_res=self.wiki_search_cache[keyword]
            )
        else:
            logger.debug("Not found in cache. Use web search.")
            for idx in range(self.max_rerun):
                try:
                    search_res = self.wiki_tool.run(keyword)
                    self.wiki_search_cache[keyword] = search_res
                    with open(self.cache_dir, "wb") as f:
                        pickle.dump(self.wiki_search_cache, f)
                    return WikiSearchRes(search_content=keyword, search_res=search_res)
                except:
                    logger.debug(traceback.format_exc())
                    logger.warning(
                        "Wiki search try: {} / {} failed.".format(idx, self.max_rerun)
                    )
                    time.sleep(1)
                    continue
        logger.error("Wiki search failed.")
        return WikiSearchRes(search_content=None, search_res=None)


class LLaMA2OutputCleaner:
    def __init__(self, is_llama_2) -> None:
        self.is_llama_2 = is_llama_2

    def clean(self, input_str: str) -> str:
        if self.is_llama_2:
            return input_str.split("\n\n")[0].strip().rstrip()
        return input_str


def pre_parse_head(s: str, all_num: int):
    s = s.strip().rstrip()
    if s.lower().startswith("The answer is ".lower()):
        s = s[len("The answer is ") :]
    s = s.split(" ")[0]
    if not (s.startswith("(") and s.endswith(")")):
        return None, False
    s = s[1:-1]
    if len(s) > 1:
        return None, False
    num_s = ord(s) - ord("a")
    if not (num_s >= 0 and num_s < all_num):
        return None, False
    return num_s, True


def pre_parse(s: str, choices: list[str]):
    head_parse_res, success = pre_parse_head(s, len(choices))
    if success:
        return head_parse_res, success
    s = s.strip().rstrip()
    s = s.lower()
    s = s.split("answer is")[-1]
    processed_choices = [f"({chr(ord('a')+idx)}) {c}" for idx, c in enumerate(choices)]
    # logger.debug(processed_choices)
    for idx, c in enumerate(processed_choices):
        if c.lower() in s:
            return idx, True
        if f"({chr(ord('a')+idx)})" in s:
            return idx, True
        if f"{chr(ord('a')+idx)})" in s:
            return idx, True
    return None, False


def get_qa_format(question, choices, search_content=None, search_res=None):
    prompt = ""
    for idx, c in enumerate(choices):
        prompt += f"({chr(ord('a')+idx)}) {c} "
    res = f"{question}\n{prompt}"
    if search_content:
        assert search_res
        res += "\nSearched from Wikipedia with content: {}, and the results are: {}".format(
            search_content, search_res
        )
        res += (
            "\nNow based on the searching results,"
            + " give me the final choice from the given choices."
            + "You mustn't add any other words and should only output your choice."
        )
    return res


def llm_call_wrapper(llm_args: LLMArgs, *args, **kwargs) -> ChatCompletion:
    curr_idx = 0
    assert len(args) == 0, args
    if llm_args.is_local:
        messages = kwargs.get("messages", None)
        max_new_tokens = kwargs.get("max_tokens", 32)
        generated_conversation: Conversation = llm_args.llm_client.generation_pipe(
            messages, max_new_tokens=max_new_tokens
        )
        generated_text = generated_conversation.generated_responses[-1]
        return ChatCompletion(
            id="",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=generated_text, role="assistant"
                    ),
                )
            ],
            created=0,
            object="chat.completion",
            model="local",
            usage=CompletionUsage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
            ),
        )
    else:
        kwargs["model"] = OPENAI_MODEL_NAME
        while curr_idx < llm_args.llm_max_rerun:
            try:
                return llm_args.llm_client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.warning(
                    "Got exception in trial {}/{}, wait for {} seconds. Details: {}".format(
                        curr_idx,
                        llm_args.llm_max_rerun,
                        (curr_idx + 1) * llm_args.llm_wait_interval,
                        json.dumps(
                            {"name": type(e).__name__, "args": e.args}, indent=2
                        ),
                    )
                )
                curr_idx += 1
                time.sleep((curr_idx + 1) * llm_args.llm_wait_interval)
        raise RuntimeError("OpenAI API call failed.")


def get_useful_result(raw_output_str: str) -> bool:
    try:
        output_json = json.loads(raw_output_str)
        useful_result = output_json["Useful"]
        assert (isinstance(useful_result, bool)) or (
            useful_result in ["true", "false"]
        ), useful_result
        if isinstance(useful_result, str):
            useful_result = useful_result == "true"
    except:
        logger.debug(
            "\033[33mUsefulAgent output error:\033[0m {}".format(traceback.format_exc())
        )
        useful_result = False
    return useful_result


def run_llm_social_agent(
    question: str,
    choices: list[str],
    usage_info: dict[str, int],
    openai_args: LLMArgs,
    wiki_searcher: WikiSearcher,
    llama2_output_cleaner: LLaMA2OutputCleaner,
    calculator: CalculatorWrapper,
) -> str:
    wiki_agent = MyOpenAIAgent(
        agent_name="WikiAgent",
        system_prompt=WIKI_AGENT_PROMPTS["system"],
        max_new_tokens=32,
    )
    calc_agent = MyOpenAIAgent(
        agent_name="CalcAgent",
        system_prompt=CALC_AGENT_PROMPTS["system"],
        max_new_tokens=32,
    )
    reasoning_agent = MyOpenAIAgent(
        agent_name="ReasoningAgent",
        system_prompt=REASONING_AGENT_PROMPTS["system"],
        max_new_tokens=2048 if openai_args.is_local else None,
    )
    useful_agent = MyOpenAIAgent(
        agent_name="UsefulAgent",
        system_prompt=USEFUL_AGENT_PROMPTS["system"],
        max_new_tokens=32,
    )
    choose_agent = MyOpenAIAgent(
        agent_name="ChooseAgent",
        system_prompt=CHOOSE_AGENT_PROMPTS["system"],
        max_new_tokens=32,
    )
    problem_prompt = get_qa_format(question, choices)
    wiki_agent_output = wiki_agent.run(
        WIKI_AGENT_PROMPTS["user"].format(problem_prompt=problem_prompt), openai_args
    )
    usage_info["prompt_tokens"] += wiki_agent_output.usage.prompt_tokens
    usage_info["completion_tokens"] += wiki_agent_output.usage.completion_tokens
    usage_info["total_tokens"] += wiki_agent_output.usage.total_tokens
    wiki_agent_output_cleaned = llama2_output_cleaner.clean(
        wiki_agent_output.choices[0].message.content
    )
    wiki_search_res = wiki_searcher.search(wiki_agent_output_cleaned)
    raw_alg_expression = calc_agent.run(
        CALC_AGENT_PROMPTS["user"].format(problem_prompt=problem_prompt), openai_args
    )
    usage_info["prompt_tokens"] += raw_alg_expression.usage.prompt_tokens
    usage_info["completion_tokens"] += raw_alg_expression.usage.completion_tokens
    usage_info["total_tokens"] += raw_alg_expression.usage.total_tokens
    raw_alg_expression_cleaned = llama2_output_cleaner.clean(
        raw_alg_expression.choices[0].message.content
    )
    alg_expression, calc_res, calc_success = calculator.run(raw_alg_expression_cleaned)
    calculator_res = CalculatorRes(
        raw_exp=raw_alg_expression,
        expression=alg_expression,
        calc_res=calc_res,
        success=calc_success,
    )
    wiki_useful = False
    calc_useful = False
    if wiki_search_res.search_content is not None:
        wiki_useful_output = useful_agent.run(
            USEFUL_AGENT_PROMPTS["user"].format(
                problem_prompt=problem_prompt,
                agent_output=json.dumps(
                    {
                        "Wikipedia Search Content": wiki_search_res.search_content,
                        "Wikipedia Search Result": wiki_search_res.search_res,
                    },
                    indent=2,
                ),
            ),
            openai_args,
        )
        usage_info["prompt_tokens"] += wiki_useful_output.usage.prompt_tokens
        usage_info["completion_tokens"] += wiki_useful_output.usage.completion_tokens
        usage_info["total_tokens"] += wiki_useful_output.usage.total_tokens
        wiki_useful_output_cleaned = llama2_output_cleaner.clean(
            wiki_useful_output.choices[0].message.content
        )
        wiki_useful = get_useful_result(wiki_useful_output_cleaned)
    if calculator_res.success:
        calc_useful_output = useful_agent.run(
            USEFUL_AGENT_PROMPTS["user"].format(
                problem_prompt=problem_prompt,
                agent_output=json.dumps(
                    {
                        "Calculator Expression": calculator_res.expression,
                        "Calculator Result": calculator_res.calc_res,
                    },
                    indent=2,
                ),
            ),
            openai_args,
        )
        usage_info["prompt_tokens"] += calc_useful_output.usage.prompt_tokens
        usage_info["completion_tokens"] += calc_useful_output.usage.completion_tokens
        usage_info["total_tokens"] += calc_useful_output.usage.total_tokens
        calc_useful_output_cleaned = llama2_output_cleaner.clean(
            calc_useful_output.choices[0].message.content
        )
        calc_useful = get_useful_result(calc_useful_output_cleaned)
    reasoning_agent_output = reasoning_agent.run(
        REASONING_AGENT_PROMPTS["user"].format(
            problem_prompt=problem_prompt,
            wiki_output=(
                (
                    "Wikipedia searching result: "
                    + json.dumps(
                        {
                            "Wikipedia Search Content": wiki_search_res.search_content,
                            "Wikipedia Search Result": wiki_search_res.search_res,
                        },
                        indent=2,
                    )
                )
                if wiki_useful
                else "Wikipedia searching result: none."
            ),
            calc_output=(
                (
                    "Calculator result: "
                    + json.dumps(
                        {
                            "Calculator Expression": calculator_res.expression,
                            "Calculator Result": calculator_res.calc_res,
                        },
                        indent=2,
                    )
                )
                if calc_useful
                else "Calculator result: none"
            ),
        ),
        openai_args,
    )
    usage_info["prompt_tokens"] += reasoning_agent_output.usage.prompt_tokens
    usage_info["completion_tokens"] += reasoning_agent_output.usage.completion_tokens
    usage_info["total_tokens"] += reasoning_agent_output.usage.total_tokens
    reasoning_agent_output_cleaned = llama2_output_cleaner.clean(
        reasoning_agent_output.choices[0].message.content
    )

    reasoning_useful = False
    reasoning_useful_output = useful_agent.run(
        USEFUL_AGENT_PROMPTS["user"].format(
            problem_prompt=problem_prompt, agent_output=reasoning_agent_output
        ),
        openai_args,
    )
    usage_info["prompt_tokens"] += reasoning_useful_output.usage.prompt_tokens
    usage_info["completion_tokens"] += reasoning_useful_output.usage.completion_tokens
    usage_info["total_tokens"] += reasoning_useful_output.usage.total_tokens
    reasoning_useful_output_cleaned = llama2_output_cleaner.clean(
        reasoning_useful_output.choices[0].message.content
    )
    reasoning_useful = get_useful_result(reasoning_useful_output_cleaned)

    final_choice = choose_agent.run(
        CHOOSE_AGENT_PROMPTS["user"].format(
            problem_prompt=problem_prompt,
            wiki_output=(
                (
                    "Wikipedia searching result: "
                    + json.dumps(
                        {
                            "Wikipedia Search Content": wiki_search_res.search_content,
                            "Wikipedia Search Result": wiki_search_res.search_res,
                        },
                        indent=2,
                    )
                )
                if wiki_useful
                else ""
            ),
            calc_output=(
                (
                    "Calculator result: "
                    + json.dumps(
                        {
                            "Calculator Expression": calculator_res.expression,
                            "Calculator Result": calculator_res.calc_res,
                        },
                        indent=2,
                    )
                )
                if calc_useful
                else ""
            ),
            reasoning_output=(
                (
                    "Reasoning agent output: "
                    + json.dumps(
                        {
                            "Step By Step Reasoning": reasoning_agent_output_cleaned,
                        },
                        indent=2,
                    )
                    + "Note that the reasoning agent output may be wrong. **DO NOT** believe in it totally!. "
                    "You should use your own judgement to decide the final answer."
                )
                if reasoning_useful
                else ""
            ),
        ),
        openai_args,
    )
    usage_info["prompt_tokens"] += final_choice.usage.prompt_tokens
    usage_info["completion_tokens"] += final_choice.usage.completion_tokens
    usage_info["total_tokens"] += final_choice.usage.total_tokens
    final_choice = final_choice.choices[0].message.content
    try:
        final_choice = json.loads(final_choice)["Answer"]
    except:
        pass
    return final_choice


def run_llm_direct(
    question: str,
    choices: list[str],
    usage_info: dict[str, int],
    openai_args: LLMArgs,
) -> str:
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant that be good at answer questions about "
        "language arts and social studies. I will give you a question and several choices."
        "You **MUST** choose one of the choices as your answer with your knowledge. "
        "You should output your answer precisely and **MUST NOT** add any other words or explanations.",
    }
    user_prompt = {
        "role": "user",
        "content": "Here are the question and choices:\n"
        + get_qa_format(question, choices)
        + "\nNow give me the final answer from the given choices.",
    }
    prompts = [system_prompt, user_prompt]
    curr_output = llm_call_wrapper(
        openai_args,
        messages=prompts,
        max_tokens=32,
        temperature=0.0,
    )
    curr_usage = curr_output.usage
    usage_info["prompt_tokens"] += curr_usage.prompt_tokens
    usage_info["completion_tokens"] += curr_usage.completion_tokens
    usage_info["total_tokens"] += curr_usage.total_tokens
    generated_text = curr_output.choices[0].message.content
    return generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gpt-3.5-turbo", "llama2", "gpt-3.5-turbo-0613"],
        default="gpt-3.5-turbo-0613",
    )
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--social_agent", action="store_true")
    parser.add_argument("--output_dir", default="./outputs/debug-output/")
    parser.add_argument(
        "--subject", choices=["social studies", "language arts", "all"], default="all"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(
        os.path.join(args.output_dir, "main_script.log"), level="DEBUG", mode="w"
    )
    logger.level("DEBUG")
    logger.info("Arguments: {}".format(json.dumps(vars(args), indent=2)))
    if args.model == "llama2":
        assert args.model_path is not None
        global WIKI_AGENT_PROMPTS, CALC_AGENT_PROMPTS, USEFUL_AGENT_PROMPTS, REASONING_AGENT_PROMPTS
        WIKI_AGENT_PROMPTS = WIKI_LLAMA_2_AGENT_PROMPTS
        CALC_AGENT_PROMPTS = CALC_LLAMA_2_AGENT_PROMPTS
        REASONING_AGENT_PROMPTS = REASONING_LLAMA_2_AGENT_PROMPTS
        USEFUL_AGENT_PROMPTS = USEFUL_LLAMA_2_AGENT_PROMPTS
    try:
        from keys import openai_key

        global OPENAI_KEY, OPENAI_MODEL_NAME
        OPENAI_KEY = openai_key
        OPENAI_MODEL_NAME = args.model
    except ImportError:
        logger.error(
            "OpenAI key not found. Create a keys.py file with the placeholder."
        )
        with open("keys.py", "w") as f:
            f.write("openai_key = '<your-openai-key>'")
        return

    social_dataset = load_dataset("socialnormdataset/social", split="test")
    if args.subject != "all":
        social_dataset = social_dataset.filter(lambda x: x["subject"] == args.subject)
    logger.info("Dataset length: {}".format(len(social_dataset)))
    output_writer = open(
        os.path.join(args.output_dir, "preds.jsonl"), "w", encoding="utf-8"
    )
    wiki_searcher = WikiSearcher()
    llama2_output_cleaner = LLaMA2OutputCleaner(is_llama_2=(args.model == "llama2"))
    calculator = CalculatorWrapper()
    if args.model in ["gpt-3.5-turbo", "llama2", "gpt-3.5-turbo-0613"]:
        llm_args = LLMArgs(
            llm_max_rerun=5 if args.model == "gpt-3.5-turbo" else 1,
            llm_wait_interval=1.0,
            llm_client=(
                OpenAI(api_key=OPENAI_KEY)
                if args.model.startswith("gpt")
                else Llama2GenerationWrapper(args.model_path)
            ),
            is_local=(args.model == "llama2"),
        )
        logger.debug(llm_args)
        usage_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        bar = tqdm(
            social_dataset,
            ascii=True,
            desc="Running evaluation, 0/0/0.0%, 0K/0K/0K",
            ncols=150,
        )
        num = 0
        correct_num = 0
        acc_by_subject = {"social studies": [0, 0, 0.0], "language arts": [0, 0, 0.0]}
        for data_item in bar:
            question, choices = data_item["question"], data_item["choices"]
            try:
                if args.social_agent:
                    generated_text = run_llm_social_agent(
                        question,
                        choices,
                        usage_info,
                        llm_args,
                        wiki_searcher,
                        llama2_output_cleaner,
                        calculator,
                    )
                else:
                    generated_text = run_llm_direct(
                        question,
                        choices,
                        usage_info,
                        llm_args,
                    )
                min_ans, success = pre_parse(generated_text, choices)
                if not success:
                    min_dis = math.inf
                    min_ans = -1
                    for c_id, c in enumerate(choices):
                        dis = Levenshtein.distance(c.lower(), generated_text.lower())
                        if dis < min_dis:
                            min_dis = dis
                            min_ans = c_id
                curr_correct = min_ans == data_item["answer_idx"]
                prediction_record = {
                    "question": question,
                    "choices": choices,
                    "answer_idx": data_item["answer_idx"],
                    "grade": data_item["grade"],
                    "skill": data_item["skill"],
                    "generation_success": True,
                    "generated_text": generated_text,
                    "parsed_success": success,
                    "parsed_answer": min_ans,
                    "correct": curr_correct,
                }
            except KeyboardInterrupt:
                logger.error("Keyboard interrupt.")
                exit(0)
            except:
                logger.error(
                    "OpenAI API call failed. Traceback: {}".format(
                        traceback.format_exc()
                    )
                )
                prediction_record = {
                    "question": question,
                    "choices": choices,
                    "answer_idx": data_item["answer_idx"],
                    "grade": data_item["grade"],
                    "skill": data_item["skill"],
                    "subject": data_item["subject"],
                    "generation_success": False,
                    "generated_text": None,
                    "parsed_success": None,
                    "parsed_answer": None,
                    "correct": None,
                }
                curr_correct = False
            num += 1
            correct_num += 1 if curr_correct else 0
            acc_by_subject[data_item["subject"]][0] += 1
            acc_by_subject[data_item["subject"]][1] += 1 if curr_correct else 0
            bar.set_description(
                "Running evaluation, {}/{}/{:.1f}%, {:.2f}K/{:.2f}K/{:.2f}K".format(
                    correct_num,
                    num,
                    correct_num / num * 100,
                    usage_info["prompt_tokens"] / 1000,
                    usage_info["completion_tokens"] / 1000,
                    usage_info["total_tokens"] / 1000,
                )
            )
            logger.debug(
                "Running evaluation {}/{}/{}, correct rate: {:.2f}%, token cosumed {:.2f}K/{:.2f}K/{:.2f}K".format(
                    correct_num,
                    num,
                    len(social_dataset),
                    correct_num / num * 100,
                    usage_info["prompt_tokens"] / 1000,
                    usage_info["completion_tokens"] / 1000,
                    usage_info["total_tokens"] / 1000,
                )
            )
            output_writer.write(
                json.dumps(prediction_record, ensure_ascii=False) + "\n"
            )
            output_writer.flush()
        output_writer.close()
        bar.close()
        logger.info(
            "Evaluation finished. "
            + "Correct rate: {:.2f}%. Token cosumed {:.2f}K/{:.2f}K/{:.2f}K".format(
                correct_num / num * 100,
                usage_info["prompt_tokens"] / 1000,
                usage_info["completion_tokens"] / 1000,
                usage_info["total_tokens"] / 1000,
            )
        )
        for subject, acc in acc_by_subject.items():
            if acc[0] == 0:
                assert acc[1] == 0
                acc[0] = 1
            logger.info(
                "Subject: {}, total: {}, correct: {}, acc: {:.2f}%".format(
                    subject, acc[0], acc[1], acc[1] / acc[0] * 100
                )
            )
    else:
        raise NotImplementedError("Model not implemented.")


if __name__ == "__main__":
    main()
