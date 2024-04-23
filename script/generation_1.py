import pandas as pd
from utils import *
from tqdm import tqdm
import regex
import traceback
from sentence_transformers import SentenceTransformer, util
import argparse
import os
import json

import lookup_utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prompt_formatting(
    generation_prompt,
    deployment_name,
    provider,
    doc,
    seed_file,
    topics_list,
    context_len,
    verbose,
    max_top_len=100,
):
    """
    Format prompt to include document and seed topics
    Handle cases where prompt is too long
    - generation_prompt: Prompt for topic generation
    - deployment_name: Model to run generation with (e.g., 'gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct')
    - provider: Provider to use for API call ('openai', 'perplexity.ai', 'together.ai')
    - doc: Document to include in prompt
    - seed_file: File to read seed topics from
    - topics_list: List of topics generated from previous iteration
    - context_len: Max context length for model (deployment_name)
    - verbose: Whether to print out results
    - max_top_len: Max length of topics to include in prompt (Modify if necessary)
    """
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    # Format seed topics to include manually written topics + previously generated topics
    topic_str = open(seed_file, "r").read() + "\n" + "\n".join(topics_list)

    # Calculate length of document, seed topics, and prompt ----
    doc_len = num_tokens_from_messages(doc, deployment_name)
    prompt_len = num_tokens_from_messages(generation_prompt, deployment_name)
    topic_len = num_tokens_from_messages(topic_str, deployment_name)
    total_len = prompt_len + doc_len + topic_len

    # Handle cases where prompt is too long ----
    if total_len > context_len:
        # Truncate document if too long
        if doc_len > (context_len - prompt_len - max_top_len):
            if verbose:
                print(f"Document is too long ({doc_len} tokens). Truncating...")
            doc = truncating(doc, context_len - prompt_len - max_top_len)
            prompt = generation_prompt.format(Document=doc, Topics=topic_str)

        # Truncate topic list to only include topics that are most similar to document
        # Determined by cosine similarity between topic string & document embedding
        else:
            if verbose:
                print(f"Too many topics ({topic_len} tokens). Pruning...")
            cos_sim = {}  # topic: cosine similarity w/ document
            doc_emb = sbert.encode(doc, convert_to_tensor=True)
            for top in topics_list:
                top_emb = sbert.encode(top, convert_to_tensor=True)
                cos_sim[top] = util.cos_sim(top_emb, doc_emb)
            sim_topics = sorted(cos_sim, key=cos_sim.get, reverse=True)

            max_top_len = context_len - prompt_len - doc_len
            seed_len, seed_str = 0, ""
            while seed_len < max_top_len and len(sim_topics) > 0:
                new_seed = sim_topics.pop(0)
                if (
                    seed_len
                    + num_tokens_from_messages(new_seed + "\n", deployment_name)
                    > max_top_len
                ):
                    break
                else:
                    seed_str += new_seed + "\n"
                    seed_len += num_tokens_from_messages(seed_str, deployment_name)
            prompt = generation_prompt.format(Document=doc, Topics=seed_str)
    else:
        prompt = generation_prompt.format(Document=doc, Topics=topic_str)
    return prompt


def append_to_json(data, file_path):
    with open(file_path, 'a') as json_file:
        json.dump(data, json_file)
        json_file.write('\n')


def overwrite_json(file_path):
    open(file_path, 'w').close()


def generate_topics(
    topics_root,
    topics_list,
    context_len,
    doc_ids,
    doc_cpcs,
    docs,
    seed_file,
    cont_out_file,
    deployment_name,
    provider,
    generation_prompt,
    temperature,
    max_tokens,
    top_p,
    verbose,
    early_stop=100,
):
    """
    Generate topics from documents using LLMs
    - topics_root, topics_list: Tree and list of topics generated from previous iteration
    - context_len: Max length of prompt
    - docs: List of documents to generate topics from
    - seed_file: File to read seed topics from
    - deployment_name: Model to run generation with (e.g., 'gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)
    - provider: Provider to use for API call ('openai', 'perplexity.ai', 'together.ai')
    - generation_prompt: Prompt to generate topics with
    - verbose: Whether to print out results
    - early_stop: Threshold for topic drought (Modify if necessary)
    """
    top_emb = {}
    responses = []
    running_dups = 0
    topic_format = regex.compile("^\[(\d+)\] ([\w\s]+):(.+)")

    overwrite_json(cont_out_file)

    for i, doc in enumerate(tqdm(docs)):
        prompt = prompt_formatting(
            generation_prompt,
            deployment_name,
            provider,
            doc,
            seed_file,
            topics_list,
            context_len,
            verbose,
        )
        try:
            response = api_call(prompt, deployment_name, provider, temperature, max_tokens, top_p)
            topics = response.split("\n")
            response_topics = []
            response_output = {"id": doc_ids.iloc[i], "cpc_codes": doc_cpcs.iloc[i], "topics": []}
            for t in topics:
                t = t.strip()
                if regex.match(topic_format, t):
                    groups = regex.match(topic_format, t)
                    lvl, name, desc = (
                        int(groups[1]),
                        groups[2].strip(),
                        groups[3].strip(),
                    )
                    if lvl == 1:
                        dups = [s for s in topics_root.descendants if s.name == name]
                        is_duplicate = len(dups) > 0
                        if is_duplicate:  # Update count if topic already exists
                            dups[0].count += 1
                            running_dups += 1
                            response_topics.append(f"DUP: [{lvl}] {name}: {desc}")
                            if running_dups > early_stop:
                                return responses, topics_list, topics_root
                        else:  # Add new topic if topic doesn't exist
                            new_node = Node(
                                name=name,
                                parent=topics_root,
                                lvl=lvl,
                                count=1,
                                desc=desc,
                            )
                            topics_list.append(f"[{new_node.lvl}] {new_node.name}")
                            response_topics.append(f"NEW: [{lvl}] {name}: {desc}")
                            running_dups = 0
                        response_output["topics"].append({"duplicate": is_duplicate, "lvl": lvl, "name": name, "desc": desc})
                    else:
                        if verbose:
                            print("Lower-level topics detected. Skipping...")
            if verbose:
                print("response output:", response_output)
                print(f"Document: {i+1}")
                print(f"Topics: {response_topics}")
                print("--------------------")
            responses.append(response)
            append_to_json(response_output, cont_out_file)

        except Exception as e:
            traceback.print_exc()
            responses.append("Error")

    return responses, topics_list, topics_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment_name",
        type=str,
        help="model to run topic generation with (e.g., 'gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500, help="max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature for generation"
    )
    parser.add_argument("--top_p", type=float, default=0.0, help="top-p for generation")
    parser.add_argument(
        "--data",
        type=str,
        default="data/input/sample.jsonl",
        help="data to run generation on",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/generation_1.txt",
        help="file to read prompts from",
    )
    parser.add_argument(
        "--cont_out_file",
        type=str,
        default="data/output/cont_generation_1.jsonl",
        help="file to continuously write results to (independent of other result files)",
    )

    parser.add_argument(
        "--seed_file",
        type=str,
        default="prompt/seed_1.md",
        help="markdown file to read the seed topics from",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/generation_1.jsonl",
        help="file to write results to",
    )
    parser.add_argument(
        "--topic_file",
        type=str,
        default="data/output/generation_1.md",
        help="file to write topics to",
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="whether to print out results"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="provider to use for API call ('openai', 'perplexity.ai', 'together.ai')",
    )
    args = parser.parse_args()

    # Model configuration ----
    deployment_name, max_tokens, temperature, top_p, provider = (
        args.deployment_name,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.provider,
    )

    context = lookup_utils.get_context_length(deployment_name, provider)
    context_len = context - max_tokens

    # Load data ----
    df = pd.read_json(str(args.data), lines=True)
    doc_ids = df["id"]  # Pass this also
    doc_cpcs = df["label"]
    docs = df["text"].tolist()
    generation_prompt = open(args.prompt_file, "r").read()
    topics_root, topics_list = generate_tree(read_seed(args.seed_file))

    # Prompting ----
    responses, topics_list, topics_root = generate_topics(
        topics_root,
        topics_list,
        context_len,
        doc_ids,
        doc_cpcs,
        docs,
        args.seed_file,
        args.cont_out_file,
        deployment_name,
        provider,
        generation_prompt,
        temperature,
        max_tokens,
        top_p,
        args.verbose,
    )

    # Writing results ----
    with open(args.topic_file, "w") as f:
        print(tree_view(topics_root), file=f)

    try:
        df = df.iloc[: len(responses)]
        df["responses"] = responses
        df.to_json(args.out_file, lines=True, orient="records")
    except Exception as e:
        traceback.print_exc()
        with open(f"data/output/generation_1_backup_{deployment_name}.txt", "w") as f:
            for line in responses:
                print(line, file=f)


if __name__ == "__main__":
    main()
