from app.llm.mistral_provider import MistralProvider


def main():
    provider = MistralProvider()

    try:
        response = provider.generate(
            system_prompt=(
                "You are a concise AI engineering assistant."
            ),
            user_prompt=(
                "Explain in one sentence what RAG means."
            ),
        )

        print("\nMISTRAL RESPONSE:")
        print(response)

    except Exception as exc:
        print("\nMISTRAL ERROR:")
        print(type(exc).__name__)
        print(exc)

        print("\nERROR DETAILS:")

        for attribute in [
            "status_code",
            "message",
            "body",
            "headers",
            "response",
        ]:
            if hasattr(exc, attribute):
                print(f"{attribute}: {getattr(exc, attribute)}")


if __name__ == "__main__":
    main()