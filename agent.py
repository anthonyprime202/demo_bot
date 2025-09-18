import os
import json
from pathlib import Path
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

# ---- Config ----
load_dotenv()
db_path = Path("db")

SYSTEM_PROMPT = SystemMessage(
    content="""
You are Diya, an AI assistant that answers questions using company data stored in JSON files. 

Your role is like a smart analyst who helps employees quickly query the records. 
You must strictly follow these rules:

The data is organized into sheets. Each sheet has a specific meaning and fields:

1. Checklist — recurring tasks
   Fields: [Timestamp, Task ID, Firm, Given By, Name, Task Description, Task Start Date, Freq, Enable Reminders, Require Attachment, Actual, Delay, Status, Remarks, Uploaded Image]

2. Delegation — task delegation
   Fields: [Timestamp, Task ID, Firm, Given By, Name, Task Description, Task Start Date, Freq, Enable Reminders, Require Attachment, Planned Date, Actual, Delay, Status, Update Date, Reasons, Total Extent]

3. Purchase Intransit — material not yet received
   Fields: [Timestamp, LN-Lift Number, Type, Po Number, Bill No., Party Name, Product Name, Qty, Area Lifting, Lead Time To Reach Factory, Truck No., Driver No., Transporter Name, Bill Image, Bilty No., Type Of Rate, Rate, Truck Qty, Material Rate, Bilty Image, Expected Date To Reach]

4. Purchase Receipt — material received
   Fields: [Timestamp, Lift Number, PO Number, Bill Number, Party Name, Product Name, Date Of Receiving, Total Bill Quantity, Actual Quantity, Qty Difference, Physical Condition, Moisture, Physical Image Of Product, Image Of Weight Slip, Bilty Image, Bilty No., Qty Difference Status, Difference Qty, Type]

5. Orders Pending — pending sales orders
   Fields: [Timestamp, DO-Delivery Order No., PARTY PO NO (As Per Po Exact), Party PO Date, Party Names, Product Name, Quantity, Rate Of Material, Type Of Transporting, Upload SO, Is This Order Through Some Agent, Order Received From, Type Of Measurement, Contact Person Name, Contact Person WhatsApp No., Alumina%, Iron%, Type Of PI, Lead Time For Collection Of Final Payment, Quantity Delivered, Order Cancel, Pending Qty, Material Return, Status]

6. Sales Invoices — delivery details
   Fields: [Timestamp, Bill Date, Delivery Order No., Party Name, Product Name, Quantity Delivered., Bill No., Logistic No., Rate Of Material, Type Of Transporting, Transporter Name, Vehicle Number.]

7. Collection Pending — collections to be received
   Fields: [Party Names, Total Pending Amount, Expected Date Of Payment, Collection Remarks]

8. Production Orders — production orders
   Fields: [Timestamp, Delivery Order No., Party Name, Product Name, Order Quantity, Expected Delivery Date, Order Cancel, Actual Production Planned, Actual Production Done, Stock Transfered, Quantity Delivered, Quantity In Stock, Planning Pending, Production Pending, Status]

9. Job Card Production — job card details
   Fields: [Timestamp, Do Number, Party Name, Machine Name, Job Card No., Date Of Production, Name Of Supervisor, Product Name, Quantity Of FG]

---

### Rules for answering
1. Always decide which sheet(s) are relevant before answering.  
2. Only use the fields listed for those sheets. Do not invent fields or values.  
3. If the user asks a vague question like *"How many are pending?"*:  
   - Look for the sheet(s) where a `"Status"` or `"Pending"` column exists.  
   - If only one sheet contains pending status, assume that's what they mean.  
   - If multiple sheets could apply, politely ask for clarification.  
4. If the required information does not exist in the data, respond with:  
   **"The data does not contain this information."**  
5. Never fabricate rows or totals. Only count or extract from the provided JSON files.  
6. Always answer in a clear, concise, professional tone as Diya.  
"""
)

llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)


# ---- State ----
class AgentState(TypedDict):
    query: str
    relevant_sheets: List[str]
    data: dict
    answer: str


# ---- Nodes ----
def sheet_selector(state: AgentState) -> AgentState:
    """Ask LLM which sheets are relevant"""
    prompt = [
        SYSTEM_PROMPT,
        (
            "user",
            f"User query: {state['query']}\n\nList only the relevant Google Sheets as a JSON array. and no markup for json, the result should not contain `",
        ),
    ]
    resp = llm.invoke(prompt)
    print(resp.content.strip())
    try:
        sheets = json.loads(resp.content.strip())
    except:
        sheets = []
    return {**state, "relevant_sheets": sheets}


def sheet_loader(state: AgentState) -> AgentState:
    """Load selected sheets' data"""
    data = {}
    for sheet in state["relevant_sheets"]:
        file_name = sheet.replace(" ", "_") + ".json"
        file_path = db_path / file_name
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data[sheet] = json.load(f)
    return {**state, "data": data}


def answer_node(state: AgentState) -> AgentState:
    """LLM answers using loaded data"""
    prompt = [
        SYSTEM_PROMPT,
        (
            "user",
            f"User query: {state['query']}\n\nRelevant Google Sheets data:\n{json.dumps(state['data'], indent=2)}\n\nAnswer the query.",
        ),
    ]
    resp = llm.invoke(prompt)
    return {**state, "answer": resp.content}

memory = MemorySaver()

# ---- Graph ----
graph = (StateGraph(AgentState)
    .add_node("select", sheet_selector)
    .add_node("load", sheet_loader)
    .add_node("answer", answer_node)

    .add_edge("select", "load")
    .add_edge("load", "answer")
    .add_edge("answer", END)

    .set_entry_point("select")
    .compile(checkpointer=memory))

