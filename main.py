import asyncio
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.live import AsyncSession

from functions.storeAddress import (
    add_complaint,
    add_complaint_tool,
    check_for_complaint,
    check_for_complaint_tool,
    get_complaint_details,
    get_complaint_details_tool,
)

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
                    if name == "check_for_complaint":
                        if not args:
                            print("Missing required parameter 'name'")
                            continue
                        argname = args["name"]
                        boolean = check_for_complaint(argname)
                        function_responses.append(
                            {
                                "name": "check_for_complaint",
                                "response": {"exists": boolean},
                                "id": call_id,
                            }
                        )
                    elif name == "add_complaint":
                        if not args:
                            print("Missing required parameters 'name' and 'address'")
                            continue
                        argname = args["name"]
                        argaddress = args["address"]

                        add_complaint(argname, argaddress)
                        response = f"Stored the address of {argname} as {argaddress}"
                        function_responses.append(
                            {
                                "name": "add_complaint",
                                "response": {"response": response},
                                "id": call_id,
                            }
                        )
                    elif name == "get_complaint_details":
                        if not args:
                            print("Missing required parameter 'name'")
                            continue
                        argname = args["name"]
                        address = get_complaint_details(argname)

                        function_responses.append(
                            {
                                "name": "get_complaint_details",
                                "response": {"name": argname, "address": address},
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
            tools=[
                types.Tool(
                    function_declarations=[
                        add_complaint_tool,
                        check_for_complaint_tool,
                        get_complaint_details_tool,
                    ]
                )
            ],
            system_instruction=types.Content(
                parts=[
                    types.Part(
                        text="You are a Customer Representative named Om"
                        ", Check for complaints and help store the names and addresses of customers, "
                        "who want to register a complaint, and help check if a complaint already exists."
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
