from agent.agent import Agent


def main():

    agent = Agent()

    print("\n" + "=" * 60)
    print("NexaCore Company Information Agent")
    print("=" * 60)

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        try:

            answer = agent.run(user_input)

            print(f"\nAgent: {answer}")

        except Exception as e:

            print(f"\nError: {e}")


if __name__ == "__main__":
    main()