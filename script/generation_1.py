import pandas as pd
from utils import *
from tqdm import tqdm
import regex
import traceback
from sentence_transformers import SentenceTransformer, util
import argparse
import os

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
    - deployment_name: Model to run generation with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct')
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


def self_ref_prompt_formatting(
    self_ref_prompt,
    deployment_name,
    provider,
    doc,
    topic,
    topics_list,
    context_len,
    verbose,
    max_top_len=100,
):
    """
    Format prompt to include document and seed topics
    Handle cases where prompt is too long
    - generation_prompt: Prompt for topic generation
    - deployment_name: Model to run generation with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct')
    - provider: Provider to use for API call ('openai', 'perplexity.ai', 'together.ai')
    - doc: Document to include in prompt
    - seed_file: File to read seed topics from
    - topics_list: List of topics generated from previous iteration
    - context_len: Max context length for model (deployment_name)
    - verbose: Whether to print out results
    - max_top_len: Max length of topics to include in prompt (Modify if necessary)
    """
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # Calculate length of document, seed topics, and prompt ----
    doc_len = num_tokens_from_messages(doc, deployment_name)
    prompt_len = num_tokens_from_messages(self_ref_prompt, deployment_name)
    topic_len = num_tokens_from_messages(topic, deployment_name)
    total_len = prompt_len + doc_len + topic_len

    # Handle cases where prompt is too long ----
    if total_len > context_len:
        # Truncate document if too long
        if doc_len > (context_len - prompt_len - max_top_len):
            if verbose:
                print(f"Document is too long ({doc_len} tokens). Truncating...")
            doc = truncating(doc, context_len - prompt_len - max_top_len)
            prompt = self_ref_prompt.format(Document=doc, Topic=topic)

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
            prompt = self_ref_prompt.format(Document=doc, Topic=seed_str)
    else:
        prompt = self_ref_prompt.format(Document=doc, Topic=topic)
    return prompt


def extract_topic_data(topic, topic_format, verbose):
    topic = topic.strip()
    groups = regex.match(topic_format, topic)

    if groups is None:
        # No topics in correct format
        return None

    lvl, name, desc = (
        int(groups[1]),
        groups[2].strip(),
        groups[3].strip(),
    )

    if lvl != 1:
        if verbose:
            print("Lower-level topics detected. Skipping...")
        return None

    return lvl, name, desc


def check_for_duplicates(name, topics_root, running_dups):
    dups = [s for s in topics_root.descendants if s.name == name]
    is_duplicate = len(dups) > 0
    if is_duplicate:  # Update count if topic already exists
        dups[0].count += 1
        running_dups += 1
    return is_duplicate, running_dups


def generate_topics(
    topics_root,
    topics_list,
    context_len,
    docs,
    seed_file,
    deployment_name,
    provider,
    generation_prompt,
    temperature,
    max_tokens,
    top_p,
    verbose,
    early_stop=100,
    self_ref_prompt=None,
):
    """
    Generate topics from documents using LLMs
    - topics_root, topics_list: Tree and list of topics generated from previous iteration
    - context_len: Max length of prompt
    - docs: List of documents to generate topics from
    - seed_file: File to read seed topics from
    - deployment_name: Model to run generation with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)
    - provider: Provider to use for API call ('openai', 'perplexity.ai', 'together.ai')
    - generation_prompt: Prompt to generate topics with
    - verbose: Whether to print out results
    - early_stop: Threshold for topic drought (Modify if necessary)
    """
    top_emb = {}
    responses = []
    running_dups = 0
    topic_format = regex.compile("^\[(\d+)\] ([\w\s]+):(.+)")

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
            for t in topics:
                topic_data = extract_topic_data(t, topic_format, verbose)
                t = t.strip()

                # Skip topic if not in correct format
                if topic_data is None:
                    continue

                lvl, name, desc = topic_data
                # Checks for duplicate topics and increments count if necessary
                is_duplicate, running_dups = check_for_duplicates(name, topics_root, running_dups)
                # Stop early if number of duplicates in a row exceeds threshold
                if running_dups > early_stop:
                    return responses, topics_list, topics_root
                # Skip node generation step if topic is a duplicate
                elif is_duplicate:
                    continue

                # Add new topic as node
                # Apply self-refinement to generated topic if provided
                if self_ref_prompt is not None:
                    prompt = self_ref_prompt_formatting(self_ref_prompt, deployment_name, provider,
                                                        doc, t, topics_list, context_len, verbose)

                    ref_response = api_call(prompt, deployment_name, provider, temperature, max_tokens, top_p)
                    t_ref = ref_response.split("\n")[0].strip()
                    topic_data = extract_topic_data(t_ref, topic_format, verbose)

                    if topic_data is not None:
                        lvl_ref, name_ref, desc_ref = topic_data
                        # TODO: remove prints once no longer testing
                        print(f"Old:{t}")
                        print(f"New: {t_ref}")
                        print()
                        is_duplicate, running_dups = check_for_duplicates(name_ref, topics_root, running_dups)
                        if running_dups > early_stop:
                            return responses, topics_list, topics_root
                        elif is_duplicate:
                            continue
                        # Overwrite with refined topic
                        lvl, name, desc = topic_data

                new_node = Node(
                    name=name,
                    parent=topics_root,
                    lvl=lvl,
                    count=1,
                    desc=desc,
                )
                topics_list.append(f"[{new_node.lvl}] {new_node.name}")
                running_dups = 0

            if verbose:
                print(f"Document: {i+1}")
                print(f"Topics: {lvl, name, desc}")
                print("--------------------")
            responses.append(response)

        except Exception as e:
            traceback.print_exc()
            responses.append("Error")

    return responses, topics_list, topics_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment_name",
        type=str,
        help="model to run topic generation with ('gpt-4', 'gpt-35-turbo', 'mistral-7b-instruct)",
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
    parser.add_argument(
        "--self_ref_prompt_file",
        type=str,
        default=None,
        help="file to read self-refinement prompt from",
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
    context = 4096
    if deployment_name == "gpt-35-turbo":
        deployment_name = "gpt-3.5-turbo"
    if deployment_name == "gpt-4":
        context = 8000
    context_len = context - max_tokens

    # Load data ----
    df = pd.read_json(str(args.data), lines=True)
    docs = df["text"].tolist()
    generation_prompt = open(args.prompt_file, "r").read()
    topics_root, topics_list = generate_tree(read_seed(args.seed_file))
    self_ref_prompt = open(args.self_ref_prompt_file, "r").read()

    # Prompting ----
    responses, topics_list, topics_root = generate_topics(
        topics_root,
        topics_list,
        context_len,
        docs,
        args.seed_file,
        deployment_name,
        provider,
        generation_prompt,
        temperature,
        max_tokens,
        top_p,
        args.verbose,
        self_ref_prompt=self_ref_prompt,
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
