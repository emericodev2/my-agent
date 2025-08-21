import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

def chatbot_node(state: State):
    """Main chatbot logic"""
    system_prompt = SystemMessage(content="You are a helpful assistant.")
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Create and compile the graph
workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

# Export the compiled app
app = workflow.compile()