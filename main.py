from pydantic_ai import Agent
from dotenv import load_dotenv
import asyncio


load_dotenv()

agent =Agent('openai-chat:gpt-4o', system_prompt="You are a helpful assistant that answers questions about the world. be super verbose")


async def main():
        async with agent.run_stream("Tell me about tokyo japan?") as result:
            async for message in result.stream_text():
                print(message)


asyncio.run(main())