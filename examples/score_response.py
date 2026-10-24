"""Score a supplied response without loading a model or dataset."""

import argparse
import json
import sys

from fi_campaign.scoring import SCORER_VERSION, score_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("gsm8k", "squad2"))
    gold = parser.add_mutually_exclusive_group(required=True)
    gold.add_argument("--gold", action="append", help="Accepted answer; repeat for alternatives.")
    gold.add_argument("--unanswerable", action="store_true", help="SQuAD2 no-answer gold.")
    parser.add_argument("--response", help="Response text; if omitted, read standard input.")
    args = parser.parse_args()
    if args.unanswerable and args.task != "squad2":
        parser.error("--unanswerable is only valid for squad2")
    response = args.response if args.response is not None else sys.stdin.read()
    answers = [""] if args.unanswerable else args.gold
    result = score_output(response, answers, args.task)
    print(json.dumps({"scorer_version": SCORER_VERSION, **result}, indent=2))


if __name__ == "__main__":
    main()
