# main_v2.py
import time
import re
import os
import certifi
import logging
import traceback
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pymongo import MongoClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.models import VectorizedQuery

app = FastAPI()

# Mount the static directory for serving HTML, CSS, JS, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration Details ---

# Azure and OpenAI configuration details
service_name = "teams-center-search-service-dev"
search_key = "4RUm7fIyJJ5GDBMX9fhV6CrGjOfKv817NKbzwnEUMPAzSeAAC5fJ"
endpoint = "https://teams-center-search-service-dev.search.windows.net"
tc_index_name = "teams-center-index-multi-application"
ir_index_name = "ir_index"


azure_openai_endpoint = "https://teams-center-azure-openai.openai.azure.com/"
azure_openai_key = "255e75cd9f2643faa9db9d073d65bce1"
azure_openai_deployment_name = "teams-center-gpt-4-o"
azure_openai_embedding_deployment = "teams-center-embedding-ADA-002"
embedding_model_name = "text-embedding-ada-002"
azure_openai_api_version = "2024-06-01"

# MongoDB (Cosmos DB) connection details
COSMOS_CONNECTION_STRING = "mongodb://cosmon-adt-02:hh4eYR3vLjeDzZzC0WseQD4yDHRBprAffa2RKpsPfXj258QZqazZDR4M54QP4a8PT7j25yr29GQXACDblQIwHQ==@cosmon-adt-02.mongo.cosmos.azure.com:10255/?ssl=true&retrywrites=false&replicaSet=globaldb&maxIdleTimeMS=120000&appName=@cosmon-adt-02@"
COSMOS_ACCOUNT_ENDPOINT = 'https://cosmon-adt-02.documents.azure.com:443/'
COSMOS_ACCOUNT_KEY = 'hh4eYR3vLjeDzZzC0WseQD4yDHRBprAffa2RKpsPfXj258QZqazZDR4M54QP4a8PT7j25yr29GQXACDblQIwHQ=='

# --- Initialize Azure and OpenAI Clients ---

# Credentials and clients
openai_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(openai_credential, "https://cognitiveservices.azure.com/.default")

embedding_client = AzureOpenAI(
    azure_deployment=azure_openai_embedding_deployment,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    azure_ad_token_provider=token_provider if not azure_openai_key else None
)

search_credential = AzureKeyCredential(search_key)
tc_search_client = SearchClient(endpoint=endpoint, index_name=tc_index_name, credential=search_credential)
ir_search_client = SearchClient(endpoint=endpoint, index_name=ir_index_name, credential=search_credential)


openai_client = AzureOpenAI(
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint,
)

# --- Initialize MongoDB Client ---

mongo_client = MongoClient(COSMOS_CONNECTION_STRING, tlsCAFile=certifi.where())  # to make SSL work on Mac
db = mongo_client["ir_tc2406_db1"]
ir_collection = db["ir_coll"]

# --- In-Memory Session Store ---
# For production, consider using Redis or another persistent store
memory_store: Dict[str, Dict[str, Any]] = {}

# --- Chatbot Functions ---

def dict_to_markdown_table(data: Dict[str, Any]) -> str:
    """
    Converts a dictionary to a Markdown-formatted table with optional highlighting.
    """
    if not isinstance(data, dict):
        return "Invalid data format."

    headers = ["Field", "Value"]
    separator = ["---", "---"]
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(separator) + " |\n"

    for key, value in data.items():
        # Highlight 'ir_status' and 'ir_priority' with emojis or bold
        if key == 'ir_status':
            if str(value).lower() == 'closed':
                value = f"**{value}**"
            elif str(value).lower() == 'open':
                value = f"**{value}**"
            # Add more statuses as needed
        elif key == 'ir_priority':
            # Example: Highlight priority
            value = f"**{value}**"
        table += f"| {key} | {value} |\n"

    return table

def get_records_by_ir_id(ir_id: str) -> Any:
    """
    Retrieves records from the 'ir_coll' collection in MongoDB based on the 'ir_id'.
    """
    try:
        ir_id_int = int(ir_id)
    except ValueError:
        return "Invalid ticket number. It must be an integer."

    records = list(ir_collection.find({"ir_id": ir_id_int}))

    if not records:
        return f"No records found for ticket number {ir_id}"

    for record in records:
        record.pop('_id', None)

    return records

def encode(content: str) -> Optional[List[float]]:
    """
    Generates embeddings for the given content using OpenAI's embedding model.
    """
    try:
        content_response = embedding_client.embeddings.create(input=content, model=embedding_model_name)
        embedding = content_response.to_dict().get("data", [])[0].get("embedding")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

def tc_search_index(query: str) -> List[Any]:
    """
    Searches the Azure Search index using vector embeddings.
    """
    embedding = encode(query)
    if not embedding:
        return []
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="embedding")
    try:
        results = tc_search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["topic_id", "folder_name", "content", "url", "version", "title"],
            top=1
        )
        results_list = list(results)
        return results_list  # Return list instead of iterator for easier handling
    except Exception as e:
        logger.error(f"Error searching index: {e}")
        return []
    
#--------------------------------------------------------------------------------------------
def ir_search_index(query: str) -> List[Any]:
    """
    Searches the Azure Search index using text-based search on ir_short_description.
    """
    try:
        results = ir_search_client.search(
            search_text=query,
            search_fields=["ir_short_description"],
            select=["id", "ir_id", "ir_short_description", "ir_activity_description", "ir_status", "ir_priority", "ir_family"],
            top=3
        )
        results_list = list(results)
        return results_list
    except Exception as e:
        print(f"Error searching index: {e}")
        return []

#------------------------------------------------------------------------------

def detect_intent(user_input: str) -> str:
    """
    Detects the intent of the user's input using OpenAI's chat completions.
    """
    examples = """
    Intent: help_request
    - Can you help me with my product?
    - I need assistance with my software.
    - How can I get support for this issue?

    Intent: ticket_request
    - What's the status of my ticket?
    - Can you check my ticket number ABC123?
    - Where is my ticket in the process?

    Intent: general_chitchat
    - Tell me a joke.
    - Hi, how are you?
    - What's the temperature in Noida, India?
    - Hello!

    Intent: feedback_positive
    - Yes, this solution works for me.
    - I'm satisfied with the solution.
    - Everything's fine now.

    Intent: feedback_negative
    - No, this didn't help.
    - I'm not satisfied with the solution.
    - The problem is still there.
    """

    try:
        response = openai_client.chat.completions.create(
            model=azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Based on the following examples, determine the intent of the user's message."},
                {"role": "user", "content": f"{examples}\n\nUser Input: {user_input}"}
            ]
        )
        intent = response.choices[0].message.content.strip()
        return intent
    except Exception as e:
        logger.error(f"Error detecting intent: {e}")
        return "unknown_intent"

#------------------------------------------------------------------------------
def handle_help_request(user_input: str, session_id: str) -> str:
    memory_store[session_id]['user_issue'] = user_input

    # Search both indexes
    tc_results = tc_search_index(user_input)
    ir_results = ir_search_index(user_input)

    context_info = ""
    ir_info = ""
    tc_info = ""

    if tc_results:
        top_tc_result = tc_results[0]
        tc_info = f"""
        Topic Folder: {top_tc_result.get('folder_name', 'N/A')}
        Topic ID: {top_tc_result.get('topic_id', 'N/A')}
        Title: {top_tc_result.get('title', 'N/A')}
        """

    if ir_results:
        top_ir_result = ir_results[0]
        ir_info = f"""
        IR ID: {top_ir_result.get('ir_id', 'N/A')}
        Short Description: {top_ir_result.get('ir_short_description', 'N/A')}
        Status: {top_ir_result.get('ir_status', 'N/A')}
        Priority: {top_ir_result.get('ir_priority', 'N/A')}
        """

    if tc_results and ir_results:
        ir_status = ir_results[0].get('ir_status', '').lower()
        if ir_status == 'closed':
            context_info = f"TC Information:\n{tc_info}\n\nIR Information:\n{ir_info}"
        else:  # ir_status is open
            context_info = f"TC Information:\n{tc_info}"
    elif tc_results and not ir_results:
        context_info = f"TC Information:\n{tc_info}"
    elif not tc_results and ir_results:
        ir_status = ir_results[0].get('ir_status', '').lower()
        if ir_status == 'open':
            context_info = f"IR Information:\n{ir_info}"
        else:
            context_info = "No relevant information found in the indexes."
    else:
        context_info = "No relevant information found in the indexes."

    try:
        response = openai_client.chat.completions.create(
            model=azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": f"Use the following information to generate an answer based on the content:\n\nUser Issue: {user_input}\n\n{context_info}"},
                {"role": "user", "content": user_input}
            ]
        )
        answer = response.choices[0].message.content.strip()

        # Prepare the bot response based on the results
        if tc_results and ir_results:
            ir_status = ir_results[0].get('ir_status', '').lower()
            ir_id = ir_results[0].get('ir_id', 'N/A')
            if ir_status == 'closed':
                bot_response = f"{answer}\n\n**Source**: [{top_tc_result['title']}]({top_tc_result.get('url', 'N/A')})\n**Reference**: IR Ticket {ir_id} (Closed)"
            else:
                bot_response = f"{answer}\n\n**Source**: [{top_tc_result['title']}]({top_tc_result.get('url', 'N/A')})\n**Note**: There is an open IR Ticket {ir_id} related to this issue."
        elif tc_results and not ir_results:
            bot_response = f"{answer}\n\n**Source**: [{top_tc_result['title']}]({top_tc_result.get('url', 'N/A')})"
        elif not tc_results and ir_results:
            ir_status = ir_results[0].get('ir_status', '').lower()
            ir_id = ir_results[0].get('ir_id', 'N/A')
            ir_short_description = ir_results[0].get('ir_short_description', 'N/A')
            if ir_status == 'open':
                bot_response = f"I found an open IR ticket related to your issue:\n\nIR Ticket: {ir_id}\nDescription: {ir_short_description}\n\nOur team is actively working on this issue. In the meantime, {answer}"
            else:
                bot_response = answer
        else:
            bot_response = answer

        memory_store[session_id]['source_url'] = top_tc_result.get('url', 'N/A') if tc_results else 'N/A'

        return bot_response
    except Exception as e:
        logger.error(f"Error handling help request: {e}")
        return "I'm sorry, I encountered an error while generating a response."
    
#------------------------------------------------------------------------------

def handle_chitchat(user_input: str) -> str:
    """
    Handles general chitchat using OpenAI's chat completions.
    """
    try:
        response = openai_client.chat.completions.create(
            model=azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful and friendly assistant. Respond to the user's message in a conversational manner."},
                {"role": "user", "content": user_input}
            ]
        )
        chitchat_response = response.choices[0].message.content.strip()
        return chitchat_response
    except Exception as e:
        logger.error(f"Error handling chitchat: {e}")
        return "I'm sorry, I couldn't process that."

def handle_product_version(user_input: str, session_id: str) -> str:
    """
    Initiates the process by asking for product name and version number.
    """
    follow_up_question = "Could you please provide the product name and version number?"
    memory_store[session_id]['last_question'] = follow_up_question
    return follow_up_question

def store_product_version(user_input: str, session_id: str) -> str:
    """
    Stores the product name and version number provided by the user.
    """
    memory_store[session_id]['product_version'] = user_input
    follow_up_question = "Could you please describe your problem in detail?"
    memory_store[session_id]['last_question'] = follow_up_question
    return f"Thank you for providing the product name and version number. {follow_up_question}"

#--------------------------------------------------------------------

def store_problem_description(user_input: str, session_id: str) -> str:
    memory_store[session_id]['problem_description'] = user_input

    # Search both indexes
    tc_results = tc_search_index(user_input)
    ir_results = ir_search_index(user_input)
    print("tc_result", tc_results)
    print("ir_results", ir_results)

    context_info = ""
    ir_info = ""
    tc_info = ""

    if tc_results:
        top_tc_result = tc_results[0]
        tc_info = f"""
        Topic Folder: {top_tc_result.get('folder_name', 'N/A')}
        Topic ID: {top_tc_result.get('topic_id', 'N/A')}
        Content: {top_tc_result.get('content', 'N/A')}
        Source: {top_tc_result.get('url', 'N/A')}
        Version: {top_tc_result.get('version', 'N/A')}
        Title: {top_tc_result.get('title', 'N/A')}
        """

    if ir_results:
        top_ir_result = ir_results[0]
        ir_info = f"""
        IR ID: {top_ir_result.get('ir_id', 'N/A')}
        Short Description: {top_ir_result.get('ir_short_description', 'N/A')}
        Activity Description: {top_ir_result.get('ir_activity_description', 'N/A')}
        Family: {top_ir_result.get('ir_family', 'N/A')}
        Status: {top_ir_result.get('ir_status', 'N/A')}
        Priority: {top_ir_result.get('ir_priority', 'N/A')}
        """

    if tc_results and ir_results:
        ir_status = ir_results[0].get('ir_status', '').lower()
        if ir_status == 'closed':
            context_info = f"TC Information:\n{tc_info}\n\nIR Information:\n{ir_info}"
        else:  # ir_status is open
            context_info = f"TC Information:\n{tc_info}"
    elif tc_results and not ir_results:
        context_info = f"TC Information:\n{tc_info}"
    elif not tc_results and ir_results:
        ir_status = ir_results[0].get('ir_status', '').lower()
        if ir_status != 'closed':
            context_info = f"IR Information:\n{ir_info}"
        else:
            context_info = "No relevant information found in the indexes."
    else:
        context_info = "No relevant information found in the indexes."

    # Send the detailed problem, product name, and version to the LLM
    problem_details = f"Product: {memory_store[session_id].get('product_version')}\nProblem: {user_input}\n\nContext Information: {context_info}"

    try:
        # LLM response
        response = openai_client.chat.completions.create(
            model=azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": f"Assist the user with the following problem:\n\n{problem_details}"},
                {"role": "user", "content": user_input}
            ]
        )
        solution = response.choices[0].message.content.strip()

        # Prepare the bot response based on the results
        if tc_results and ir_results:
            ir_status = ir_results[0].get('ir_status', '').lower()
            ir_id = ir_results[0].get('ir_id', 'N/A')
            if ir_status == 'closed':
                bot_response = f"{solution}\n\n**Source**: [{top_tc_result['title']}]({top_tc_result['url']})\n**Reference**: IR Ticket {ir_id} (Closed)\n\nAre you satisfied with this solution?"
            else:
                bot_response = f"{solution}\n\n**Source**: [{top_tc_result['title']}]({top_tc_result['url']})\n**Note**: There is an open IR Ticket {ir_id} related to this issue.\n\nAre you satisfied with this solution?"
        elif tc_results and not ir_results:
            bot_response = f"{solution}\n\n**Source**: [{top_tc_result['title']}]({top_tc_result['url']})\n\nAre you satisfied with this solution?"
        elif not tc_results and ir_results:
            ir_status = ir_results[0].get('ir_status', '').lower()
            ir_id = ir_results[0].get('ir_id', 'N/A')
            ir_short_description = ir_results[0].get('ir_short_description', 'N/A')
            if ir_status == 'open':
                bot_response = f"I found an open IR ticket related to your issue:\n\nIR Ticket: {ir_id}\nDescription: {ir_short_description}\n\nOur team is actively working on this issue. In the meantime, {solution}\n\nAre you satisfied with this information?"
            else:
                bot_response = f"{solution}\n\nAre you satisfied with this solution?"
        else:
            bot_response = f"{solution}\n\nAre you satisfied with this solution?"

        # Store solution and ask for feedback
        memory_store[session_id]['solution'] = solution
        memory_store[session_id]['last_question'] = "Are you satisfied with this solution?"

        return bot_response
    except Exception as e:
        logger.error(f"Error storing problem description: {e}")
        return "I'm sorry, I encountered an error while processing your problem."
    
#-------------------------------------------------------------------------------

def handle_feedback(user_input: str, session_id: str) -> str:
    """
    Handles user feedback and redirects if negative.
    """
    intent = detect_intent(user_input)

    if 'feedback_positive' in intent.lower():
        memory_store[session_id]['last_question'] = None  # Clear follow-up
        return "**Great! I'm glad the solution helped.**"
    elif 'feedback_negative' in intent.lower():
        memory_store[session_id]['last_question'] = None  # Clear follow-up
        return "I'm sorry the solution didn't help. Please raise a support ticket at: [SIEMENS Support](https://support.sw.siemens.com/en-US)"
    else:
        return "I didn't catch that. Could you let me know if you're satisfied with the solution?"

def extract_ticket_number(user_input: str) -> Optional[str]:
    """
    Extracts the ticket number from the user's input.
    """
    ticket_number = re.findall(r'\b\d+\b', user_input)
    if ticket_number:
        return ticket_number[0]
    else:
        return None

def handle_ticket_number(user_input: str, session_id: str) -> str:
    """
    Handles the user's ticket number input and fetches its status.
    """
    ticket_number = extract_ticket_number(user_input)
    if ticket_number:
        records = get_records_by_ir_id(ticket_number)
        if isinstance(records, list) and records:
            # Assuming only one record per ticket_id
            record = records[0]
            table = dict_to_markdown_table(record)
            bot_response = f"**Here are the details for ticket {ticket_number}:**\n\n{table}"
            memory_store[session_id]['last_question'] = None  # Clear last question
        else:
            # This means an error or no records found
            bot_response = records  # Already a string message
            memory_store[session_id]['last_question'] = None  # Clear last question
    else:
        bot_response = "Sorry, I couldn't find a ticket number in your response. Please provide your ticket number."
    return bot_response

def handle_query(user_input: str, session_id: str) -> str:
    """
    Determines the appropriate response based on user input and session state.
    """
    intent = detect_intent(user_input)

    # Initialize session in memory_store if not present
    if session_id not in memory_store:
        memory_store[session_id] = {}

    # Handle general chitchat regardless of the flow
    if 'general_chitchat' in intent.lower():
        bot_response = handle_chitchat(user_input)
        # Resume last question if any
        if memory_store[session_id].get('last_question'):
            time.sleep(1)  # Optional: simulate delay
            bot_response += f"\n\n{memory_store[session_id]['last_question']}"
        return bot_response

    # Handle feedback during feedback phase
    if memory_store[session_id].get('last_question') == "Are you satisfied with this solution?":
        return handle_feedback(user_input, session_id)

    # Handle pending questions
    if memory_store[session_id].get('last_question'):
        last_question = memory_store[session_id]['last_question']
        if last_question == "Could you please provide the product name and version number?":
            return store_product_version(user_input, session_id)
        elif last_question == "Could you please describe your problem in detail?":
            return store_problem_description(user_input, session_id)
        elif last_question == "Please provide your ticket number.":
            return handle_ticket_number(user_input, session_id)
        else:
            return "I'm sorry, I didn't understand that."

    # No pending question, handle normally
    if 'help_request' in intent.lower():
        return handle_product_version(user_input, session_id)
    elif 'ticket_request' in intent.lower():
        # Try to extract ticket number from user input
        ticket_number = extract_ticket_number(user_input)
        if ticket_number:
            # We have a ticket number, proceed to get records
            records = get_records_by_ir_id(ticket_number)
            if isinstance(records, list) and records:
                # Records found, format the response
                record_details = "\n".join([str(record) for record in records])
                bot_response = f"**Here are the details for ticket {ticket_number}:**\n\n{record_details}"
            else:
                # This means an error or no records found
                bot_response = records
            return bot_response
        else:
            # No ticket number found, ask for it
            follow_up_question = "Please provide your ticket number."
            memory_store[session_id]['last_question'] = follow_up_question
            return "Sure! Please provide your ticket number, and I'll check the status for you."
    elif 'general_chitchat' in intent.lower():
        return handle_chitchat(user_input)
    else:
        return "I'm sorry, I couldn't understand your request. Could you please clarify?"

# --- FastAPI Endpoints ---

@app.post("/message")
async def receive_message(session_id: str = Form(...), message: str = Form(...)):
    """
    Receives a message from the frontend, processes it, and returns a response.
    """
    try:
        bot_response = handle_query(message, session_id)
        return JSONResponse(content={"message": bot_response})
    except Exception as e:
        logger.error(f"Error processing message: {e}\n{traceback.format_exc()}")
        return JSONResponse(content={"message": "An error occurred while processing your request."}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """
    Serves the frontend HTML page.
    """
    try:
        with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return HTMLResponse(content="Error loading the page.", status_code=500)
