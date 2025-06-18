"""Run the AI Assistant."""
from assistant import Assistant

def main():
    print("🤖 Simple Cohere AI Assistant")
    print("Commands: /help, /clear, /save, /summarize <text>, /quit")
    print("-" * 50)

    try:
        assistant = Assistant()
    except ValueError as e:
        print(f"❌ {e}")
        print("Get your API key from: https://dashboard.cohere.ai/")
        return

    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input == '/quit':
                print("👋 Goodbye!")
                break
            elif user_input == '/help':
                print("\nCommands:")
                print("/help - Show this help")
                print("/clear - Clear chat history")
                print("/save - Save conversation")
                print("/summarize <text> - Summarize text")
                print("/quit - Exit")
                continue
            elif user_input == '/clear':
                print(assistant.clear_history())
                continue
            elif user_input == '/save':
                print(assistant.save_chat())
                continue
            elif user_input.startswith('/summarize '):
                text = user_input[11:].strip()
                if text:
                    print(f"🤖 Summary: {assistant.summarize(text)}")
                else:
                    print("Please provide text to summarize")
                continue

            # Regular chat
            response = assistant.chat(user_input)
            print(f"🤖 Assistant: {response}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
