from llm_client import LLMClient
from tools.tool_definition import tools
from tools.tools import TOOL_REGISTRY

SYSTEM_PROMPT = """
You are the NexaCore Technologies Company Information Assistant.

Your job is to answer questions using information from the
company knowledge base.

When a question requires company-specific information,
use the search_knowledge_base tool.

IMPORTANT RULES:

1. Do not invent company policies, numbers, dates, procedures,
   benefits, or other company-specific information.

2. When the knowledge base returns context, use that context
   as the primary evidence for your answer.

3. Only state company-specific facts that are supported by
   the retrieved context.

4. If the retrieved context does not contain enough information
   to answer the question, clearly say that you could not find
   the information in the company knowledge base.

5. Do not replace missing company information with general
   knowledge or guesses.

6. When answering from retrieved information, mention the
   relevant source document and page when possible.

7. Keep answers concise and directly answer the user's question.
"""

class Agent:

    def __init__(self):
        self.llm = LLMClient(model="llama3.2:3b")
        self.tools = tools
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

    def run(self, user_input: str):

        # --------------------------------------------
        # Add user message to agent state
        # --------------------------------------------

        self.messages.append({
            "role": "user",
            "content": user_input
        })

        max_steps = 10

        for step in range(max_steps):

            response = self.llm.chat(
                messages=self.messages,
                tools=self.tools,
            )

            message = response["message"]

            tool_calls = message.get("tool_calls", [])

            # --------------------------------------------
            # No tool call = final answer
            # --------------------------------------------

            if not tool_calls:

                # Store assistant's final response
                self.messages.append(message)

                return message["content"]

            # --------------------------------------------
            # Store assistant's tool-call message
            # --------------------------------------------

            self.messages.append(message)

            for tool_call in tool_calls:

                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                print(f"\n--- Agent Step {step + 1} ---")
                print("Tool requested:", tool_name)
                print("Arguments:", arguments)

                # ----------------------------------------
                # Check whether tool exists
                # ----------------------------------------

                if tool_name not in TOOL_REGISTRY:

                    result = {
                        "error": f"Unknown tool: {tool_name}"
                    }

                else:

                    tool_info = TOOL_REGISTRY[tool_name]

                    tool_function = tool_info["function"]
                    tool_schema = tool_info["schema"]

                    try:

                        validated_arguments = tool_schema(**arguments)

                        result = tool_function(
                            **validated_arguments.model_dump()
                        )

                    except Exception as e:

                        result = {
                            "error": str(e)
                        }

                print("Tool result:", result)

                # ----------------------------------------
                # Add tool result to agent state
                # ----------------------------------------

                self.messages.append({
                    "role": "tool",
                    "content": str(result),
                })

        # --------------------------------------------
        # Maximum number of agent steps reached
        # --------------------------------------------

        raise RuntimeError(
            "Agent exceeded maximum number of steps."
        )