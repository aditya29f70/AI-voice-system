from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from app.graph.state import VoiceCallState,LeadExecutionState
from app.graph.nodes.tts import text_to_speech
from app.graph.nodes.stt import speech_to_text
from app.graph.nodes.fast_reply_llm import fast_reply_llm
from app.graph.nodes.language import detect_language
from app.graph.nodes.lead_extraction import lead_extraction
from app.graph.nodes.hot_action import hot_action
from app.graph.nodes.warm_action import warm_action
from app.graph.routing import decide_hot_warm
import os


builder1= StateGraph(VoiceCallState)
builder2= StateGraph(LeadExecutionState)

builder1.add_node('speech_to_text', speech_to_text)
builder1.add_node("detect_language", detect_language)
builder1.add_node("fast_reply_llm", fast_reply_llm)
builder1.add_node("text_to_speech", text_to_speech)

builder2.add_node("lead_extraction", lead_extraction)
builder2.add_node("hot_action", hot_action)
builder2.add_node("warm_action", warm_action)



builder1.add_edge(START, "speech_to_text")
builder1.add_edge("speech_to_text", "detect_language")
builder1.add_edge("detect_language", "fast_reply_llm")
builder1.add_edge("fast_reply_llm", "text_to_speech")
builder1.add_edge("text_to_speech",END)

builder2.add_edge(START, "lead_extraction")
builder2.add_conditional_edges("lead_extraction", decide_hot_warm)
builder2.add_edge("hot_action",END)
builder2.add_edge("warm_action", END)


graph1_checkpointer= InMemorySaver()
graph2_checkpointer= InMemorySaver()

graph1= builder1.compile(graph1_checkpointer)
graph2= builder2.compile(graph2_checkpointer)

graph1_png= graph1.get_graph().draw_mermaid_png()
graph2_png= graph2.get_graph().draw_mermaid_png()


graph1_path= os.path.join(os.getcwd(), "app\graph\graph1.png")
graph2_path= os.path.join(os.getcwd(), "app\graph\graph2.png")

with open(graph1_path, "wb") as f:
    f.write(graph1_png)

with open(graph2_path, "wb") as f:
    f.write(graph2_png)
