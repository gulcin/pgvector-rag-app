from rag import rag_query


def chat(args, model, device, tokenizer):
    print("Chat started. Type 'exit' to end the chat.")

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            break

        answer = rag_query(tokenizer=tokenizer, model=model, device=device, query=question)

        print(f"You Asked: {question}")
        print(f"Answer: {answer}")

    print("Chat ended.")
