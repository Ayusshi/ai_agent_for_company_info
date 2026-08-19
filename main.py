from agent.agent import ask_agent


def main():

    print("=" * 60)
    print("AI COMPANY ASSISTANT")
    print("=" * 60)

    print(
        "\nType 'exit' to quit."
    )

    while True:

        question = input(
            "\nYou: "
        )

        if question.lower().strip() == "exit":
            break

        try:

            answer = ask_agent(
                question=question,
                thread_id="cli-session",
            )

            print(
                f"\nAssistant: {answer}"
            )

        except Exception as error:

            print(
                f"\nError: {error}"
            )


if __name__ == "__main__":
    main()