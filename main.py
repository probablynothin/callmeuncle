import asyncio
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.live import AsyncSession

from functions.getWeather import get_weather, get_weather_tool

load_dotenv()


async def receive_data(session: AsyncSession):
    print("Gemini: ", end="")
    async for response in session.receive():
        if response.text:
            print(response.text, end="")
        elif response.tool_call:
            print("tool call received", response.tool_call)

            function_responses = []
            if response.tool_call.function_calls:
                for function_call in response.tool_call.function_calls:
                    name = function_call.name
                    args = function_call.args
                    # Extract the numeric part from Gemini's function call ID
                    call_id = function_call.id
                    if name == "get_weather":
                        if not args:
                            print("Missing required parameter 'location'")
                            continue
                        location = args["location"]
                        temperature = get_weather(location)
                        function_responses.append(
                            {
                                "name": "get_weather",
                                "response": {"temperature": temperature},
                                "id": call_id,
                            }
                        )

                    else:
                        print(f"Unknown function name: {function_call.name}")

            await session.send(function_responses)


async def main():
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={"api_version": "v1alpha"},
    )
    async with client.aio.live.connect(
        model="gemini-2.0-flash-exp",
        config=types.LiveConnectConfig(
            response_modalities=["TEXT"],
            tools=[types.Tool(function_declarations=[get_weather_tool])],
            system_instruction=types.Content(
                parts=[
                    types.Part(
                        text="You Are a Helpful Assitant named Doug , introduce yourself and help find weather information for the user"
                    )
                ]
            ),
        ),
    ) as session:
        session: AsyncSession = session
        while True:
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                await session.close()
                print("Chatbot: Goodbye!")
                break

            # Generate a response from the model
            await session.send(user_input, end_of_turn=True)
            await receive_data(session)


if __name__ == "__main__":
    asyncio.run(main())
