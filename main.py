from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import asyncio


load_dotenv()

MODEL = "openai-chat:gpt-4o"

weather_agent = Agent(
    MODEL,
    name="weather_agent",
    instructions="You report the weather for a city. Always use get_weather. Keep answers short.",
)


@weather_agent.tool_plain
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny with a temperature of 25°C."


city_guide_agent = Agent(
    MODEL,
    name="city_guide_agent",
    instructions="You are a concise city guide. Give 3 or 4 interesting facts about a place.",
)


planner = Agent(
    MODEL,
    name="planner",
    instructions=(
        "You coordinate specialist agents. "
        "Use ask_weather for weather and ask_city_guide for local facts, "
        "then combine their answers into one helpful reply."
    ),
)


@planner.tool
async def ask_weather(ctx: RunContext, city: str) -> str:
    """Ask the weather specialist about a city."""
    result = await weather_agent.run(
        f"What is the weather in {city}?",
        usage=ctx.usage,
    )
    return result.output


@planner.tool
async def ask_city_guide(ctx: RunContext, city: str) -> str:
    """Ask the city guide specialist about a place."""
    result = await city_guide_agent.run(
        f"Tell me interesting facts about {city}.",
        usage=ctx.usage,
    )
    return result.output


async def main():
    async with planner.run_stream("I'm visiting Tokyo. What should I know?") as result:
        async for message in result.stream_text():
            print(message, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
