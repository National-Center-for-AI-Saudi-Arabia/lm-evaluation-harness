
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
            [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
        )
        prompt += "Answer:"
        return prompt
    
    keys = ["A", "B", "C", "D"]
    i = keys.index("B")
    j = doc["answer"]
    # Swap the content of the two positions
    doc["choices"][i], doc["choices"][j] = doc["choices"][j], doc["choices"][i]
    return {
        "question": format_example(doc, keys),
        "choices": keys,
        "answer": i,
    }

def process_docs(dataset):
    return dataset.map(_process_doc)
