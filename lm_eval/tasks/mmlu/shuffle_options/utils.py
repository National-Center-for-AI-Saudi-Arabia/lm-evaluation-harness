
import random

def _process_doc(doc):
    def format_example(doc, keys):
        """
        Question: <prompt>
        Choices:
        A. <choice1>
        B. <choice2>
        C. <choice3>
        D. <choice4>
        Answer:
        """
        prompt = "Question: " + doc["question"] + "\nChoices:\n"
        prompt += "".join(
            [f"{key}. {choice}\n" for key, choice in sorted(zip(keys, doc["choices"]), key=lambda _: random.random())]
        )
        prompt += "Answer:"
        return prompt
    
    keys = ["A", "B", "C", "D"]
    return {
        "question": format_example(doc, keys),
        "choices": keys,
        "answer": doc["answer"],
    }

def process_docs(dataset):
    return dataset.map(_process_doc)
