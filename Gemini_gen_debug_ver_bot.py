#!/usr/bin/env python
# coding: utf-8

# ## Chatbot with Customized Tool
# ### Version support deliver to customer, for customer support

# 1) User Input: The conversation begins with the user's query. This input, along with a RunnableConfig (crucially containing the thread_id for memory management), is sent to the LangGraph agent.
# 
# 
# 2) LLM (Orchestrator): The LLM is the central brain. It receives the user's message and, importantly, has access to the ChatbotState (memory).
# 
# 
# 3) ChatbotState (Memory): This is the persistent state of the conversation. The LLM reads from it to understand context (e.g., previous turns, user preferences, results of past tool calls) and writes to it to update the conversation's progress. The thread_id from RunnableConfig tells the checkpointer which specific conversation's state to load/save.
# 
# 
# 4) Decision: Based on the user's input and the current ChatbotState, the LLM decides the next action. This is the core of its "reasoning."
# 
# 
# 5) If a Tool Call: If the LLM determines that an external action or information retrieval is needed (e.g., "find flights," "what's the weather?"), it formulates a tool call.
# 
# 
# 6) If Direct Response: If the LLM can answer the question directly from its internal knowledge or the current ChatbotState, it generates a conversational response.
# 
# 
# 7) Tool Executor: If a tool call is made, the ToolExecutor node in LangGraph takes over. It receives the tool call instructions from the LLM.
# 
# 
# 8) Invokes Specific Tool: If the LLM requested a specific action (like search_flights), the ToolExecutor calls the corresponding custom tool function. This tool then interacts with its dedicated external API.
# 
# 
# 9) Invokes General Search Tool: If the LLM requested general information (like "What is the capital of France?"), the ToolExecutor calls the TavilySearch tool. This tool then performs a web search via the Tavily API.
# Tool Output / Search Results: The results from the external APIs (whether specific tools or Tavily) are returned to the ToolExecutor.
# 
# 
# 10) ToolMessage (Output): The ToolExecutor wraps the tool's output in a ToolMessage and sends it back to the LLM. This allows the LLM to see the results of the action it requested.
# Generates Final Response: The LLM, now with the tool's output (or having decided on a direct response earlier), formulates the final, coherent answer for the user.
# 
# 
# 11) Chatbot Output: The final response is delivered to the user.
# 

# ### pre: first Populate database

# In[40]:


import os
import shutil
import sqlite3

import pandas as pd
import requests

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
# The backup lets us restart for each tutorial section
backup_file = "travel2.backup.sqlite"
overwrite = False
if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_file, "wb") as f:
        f.write(response.content)
    # Backup - we will use this to "reset" our DB in each section
    shutil.copy(local_file, backup_file)


# Convert the flights to present time for our tutorial
def update_dates(file):
    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()

    print(f"Database {file} updated with current dates.") # DEBUG
    return file


db = update_dates(local_file)


# ### Input API keys for tool defining prep (OpenAI, Tavily, Langsmith)

# In[41]:


from dotenv import load_dotenv
import os


# for llm
load_dotenv('openai_api_key.env ') 
api_key = os.getenv('OPENAI_API_KEY') 
print(f"OpenAI API Key loaded: {api_key is not None}")
print(f"First few characters: {api_key[:5]}..." if api_key else "No key found")
print('')


# for tavilySearch, allow llm to search online
load_dotenv('t_api_key.env')
t_api_key = os.getenv('TAVILY_API_KEY')
print(f"tavily API Key loaded: {t_api_key is not None}")
print(f"First few characters: {t_api_key[:5]}..." if api_key else "No key found")
print('')


# for eval
load_dotenv('Langsmith_api_k.env')
ls_api_key = os.getenv('LANGSMITH_API_KEY')
print(f"LangSmith API Key loaded: {ls_api_key is not None}")
print(f"First few characters: {ls_api_key[:5]}..." if api_key else "No key found")


# In[42]:


import re

import numpy as np
import openai
from langchain_core.tools import tool

response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-large", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


retriever = VectorStoreRetriever.from_docs(docs, openai.Client())


@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events."""
    print(f"DEBUG: lookup_policy called with query: '{query}'") # DEBUG
    docs = retriever.query(query, k=2)
    result = "\n\n".join([doc["page_content"] for doc in docs])
    print(f"DEBUG: lookup_policy returning: '{result[:100]}...'") # DEBUG: Truncate for brevity
    return result


# ### Define TOOLS
# 
# 1) __define customized tools__ for specific task
# 
# 
# 1.1 · Flight inquiries and reservation changes
# 
# 
# 1.2 · Hotel, car rental, and travel activity arrangements
# 
# 
# 1.3 · Policy inquiries and compliance checks

# In[134]:


# 1.1

import sqlite3
from datetime import date, datetime
from typing import Optional

import pytz
from langchain_core.runnables import RunnableConfig


@tool
def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments.

    Returns:
        A list of dictionaries where each dictionary contains the ticket details,
        associated flight details, and the seat assignments for each ticket belonging to the user.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    print(f"DEBUG: fetch_user_flight_information called for passenger_id: {passenger_id}") # DEBUG
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    print(f"DEBUG: fetch_user_flight_information returning {len(results)} results.") # DEBUG
    return results


@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[date | datetime] = None,
    end_time: Optional[date | datetime] = None,
    limit: int = 20,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    print(f"DEBUG: search_flights called with: departure_airport={departure_airport}, arrival_airport={arrival_airport}, start_time={start_time}, end_time={end_time}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []

    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)

    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)

    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(str(start_time)) # Convert datetime/date to string for SQL
    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(str(end_time)) # Convert datetime/date to string for SQL

    query += " LIMIT ?"
    params.append(limit)
    
    print(f"DEBUG: search_flights SQL query: {query} with params: {params}") # DEBUG
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    print(f"DEBUG: search_flights returning {len(results)} results.") # DEBUG
    return results


@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """Update the user's ticket to a new valid flight."""
    print(f"DEBUG: update_ticket_to_new_flight called with ticket_no={ticket_no}, new_flight_id={new_flight_id}") # DEBUG
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        print(f"DEBUG: update_ticket_to_new_flight returning 'Invalid new flight ID provided.'") # DEBUG
        return "Invalid new flight ID provided."
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))
    timezone = pytz.timezone("Etc/GMT-3")
    current_time = datetime.now(tz=timezone)
    departure_time = datetime.strptime(
        new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
    )
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 3600):
        print(f"DEBUG: update_ticket_to_new_flight returning 'Not permitted to reschedule...'") # DEBUG
        return f"Not permitted to reschedule to a flight that is less than 3 hours from the current time. Selected flight is at {departure_time}."

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        print(f"DEBUG: update_ticket_to_new_flight returning 'No existing ticket found...'") # DEBUG
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        print(f"DEBUG: update_ticket_to_new_flight returning 'Current signed-in passenger not owner...'") # DEBUG
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    # In a real application, you'd likely add additional checks here to enforce business logic,
    # like "does the new departure airport match the current ticket", etc.
    # While it's best to try to be *proactive* in 'type-hinting' policies to the LLM
    # it's inevitably going to get things wrong, so you **also** need to ensure your
    # API enforces valid behavior
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    print(f"DEBUG: update_ticket_to_new_flight returning 'Ticket successfully updated.'") # DEBUG
    return "Ticket successfully updated to new flight."


@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Cancel the user's ticket and remove it from the database."""
    print(f"DEBUG: cancel_ticket called with ticket_no={ticket_no}") # DEBUG
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        print(f"DEBUG: cancel_ticket returning 'No existing ticket found...'") # DEBUG
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        print(f"DEBUG: cancel_ticket returning 'Current signed-in passenger not owner...'") # DEBUG
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    print(f"DEBUG: cancel_ticket returning 'Ticket successfully cancelled.'") # DEBUG
    return "Ticket successfully cancelled."


# In[44]:


# 1.2 car rental+hotel

from datetime import date, datetime
from typing import Optional, Union


@tool
def search_car_rentals(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for car rentals based on location, name, price tier, start date, and end date.

    Args:
        location (Optional[str]): The location of the car rental. Defaults to None.
        name (Optional[str]): The name of the car rental company. Defaults to None.
        price_tier (Optional[str]): The price tier of the car rental. Defaults to None.
        start_date (Optional[Union[datetime, date]]): The start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The end date of the car rental. Defaults to None.

    Returns:
        list[dict]: A list of car rental dictionaries matching the search criteria.
    """
    print(f"DEBUG: search_car_rentals called with: location={location}, name={name}, price_tier={price_tier}, start_date={start_date}, end_date={end_date}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM car_rentals WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # For our tutorial, we will let you match on any dates and price tier.
    # (since our toy dataset doesn't have much data)
    
    print(f"DEBUG: search_car_rentals SQL query: {query} with params: {params}") # DEBUG
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()
    
    dict_results = [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]
    print(f"DEBUG: search_car_rentals returning {len(dict_results)} results.") # DEBUG
    return dict_results


@tool
def book_car_rental(rental_id: int) -> str:
    """
    Book a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to book.

    Returns:
        str: A message indicating whether the car rental was successfully booked or not.
    """
    print(f"DEBUG: book_car_rental called with rental_id: {rental_id}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        print(f"DEBUG: book_car_rental returning 'Car rental {rental_id} successfully booked.'") # DEBUG
        return f"Car rental {rental_id} successfully booked."
    else:
        conn.close()
        print(f"DEBUG: book_car_rental returning 'No car rental found with ID {rental_id}.'") # DEBUG
        return f"No car rental found with ID {rental_id}."


@tool
def update_car_rental(
    rental_id: int,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a car rental's start and end dates by its ID.

    Args:
        rental_id (int): The ID of the car rental to update.
        start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.

    Returns:
        str: A message indicating whether the car rental was successfully updated or not.
    """
    print(f"DEBUG: update_car_rental called with rental_id={rental_id}, start_date={start_date}, end_date={end_date}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if start_date:
        cursor.execute(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (str(start_date), rental_id), # Convert to string
        )
    if end_date:
        cursor.execute(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?", (str(end_date), rental_id) # Convert to string
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        print(f"DEBUG: update_car_rental returning 'Car rental {rental_id} successfully updated.'") # DEBUG
        return f"Car rental {rental_id} successfully updated."
    else:
        conn.close()
        print(f"DEBUG: update_car_rental returning 'No car rental found with ID {rental_id}.'") # DEBUG
        return f"No car rental found with ID {rental_id}."


@tool
def cancel_car_rental(rental_id: int) -> str:
    """
    Cancel a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to cancel.

    Returns:
        str: A message indicating whether the car rental was successfully cancelled or not.
    """
    print(f"DEBUG: cancel_car_rental called with rental_id: {rental_id}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        print(f"DEBUG: cancel_car_rental returning 'Car rental {rental_id} successfully cancelled.'") # DEBUG
        return f"Car rental {rental_id} successfully cancelled."
    else:
        conn.close()
        print(f"DEBUG: cancel_car_rental returning 'No car rental found with ID {rental_id}.'") # DEBUG
        return f"No car rental found with ID {rental_id}."
    
@tool
def search_hotels(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for hotels based on location, name, price tier, check-in date, and check-out date.

    Args:
        location (Optional[str]): The location of the hotel. Defaults to None.
        name (Optional[str]): The name of the hotel. Defaults to None.
        price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
        checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.

    Returns:
        list[dict]: A list of hotel dictionaries matching the search criteria.
    """
    print(f"DEBUG: search_hotels called with: location={location}, name={name}, price_tier={price_tier}, checkin_date={checkin_date}, checkout_date={checkout_date}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM hotels WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # For the sake of this tutorial, we will let you match on any dates and price tier.
    
    print(f"DEBUG: search_hotels SQL query: {query} with params: {params}") # DEBUG
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    dict_results = [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]
    print(f"DEBUG: search_hotels returning {len(dict_results)} results.") # DEBUG
    return dict_results


@tool
def book_hotel(hotel_id: int) -> str:
    """
    Book a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to book.

    Returns:
        str: A message indicating whether the hotel was successfully booked or not.
    """
    print(f"DEBUG: book_hotel called with hotel_id: {hotel_id}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        print(f"DEBUG: book_hotel returning 'Hotel {hotel_id} successfully booked.'") # DEBUG
        return f"Hotel {hotel_id} successfully booked."
    else:
        conn.close()
        print(f"DEBUG: book_hotel returning 'No hotel found with ID {hotel_id}.'") # DEBUG
        return f"No hotel found with ID {hotel_id}."


@tool
def update_hotel(
    hotel_id: int,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a hotel's check-in and check-out dates by its ID.

    Args:
        hotel_id (int): The ID of the hotel to update.
        checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.

    Returns:
        str: A message indicating whether the hotel was successfully updated or not.
    """
    print(f"DEBUG: update_hotel called with hotel_id={hotel_id}, checkin_date={checkin_date}, checkout_date={checkout_date}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if checkin_date:
        cursor.execute(
            "UPDATE hotels SET checkin_date = ? WHERE id = ?", (str(checkin_date), hotel_id) # Convert to string
        )
    if checkout_date:
        cursor.execute(
            "UPDATE hotels SET checkout_date = ? WHERE id = ?",
            (str(checkout_date), hotel_id), # Convert to string
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        print(f"DEBUG: update_hotel returning 'Hotel {hotel_id} successfully updated.'") # DEBUG
        return f"Hotel {hotel_id} successfully updated."
    else:
        conn.close()
        print(f"DEBUG: update_hotel returning 'No hotel found with ID {hotel_id}.'") # DEBUG
        return f"No hotel found with ID {hotel_id}."


@tool
def cancel_hotel(hotel_id: int) -> str:
    """
    Cancel a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to cancel.

    Returns:
        str: A message indicating whether the hotel was successfully cancelled or not.
    """
    print(f"DEBUG: cancel_hotel called with hotel_id: {hotel_id}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        print(f"DEBUG: cancel_hotel returning 'Hotel {hotel_id} successfully cancelled.'") # DEBUG
        return f"Hotel {hotel_id} successfully cancelled."
    else:
        conn.close()
        print(f"DEBUG: cancel_hotel returning 'No hotel found with ID {hotel_id}.'") # DEBUG
        return f"No hotel found with ID {hotel_id}."


# In[45]:


# 1.3 search tool

@tool
def search_trip_recommendations(
    location: Optional[str] = None,
    name: Optional[str] = None,
    keywords: Optional[str] = None,
) -> list[dict]:
    """
    Search for trip recommendations based on location, name, and keywords.

    Args:
        location (Optional[str]): The location of the trip recommendation. Defaults to None.
        name (Optional[str]): The name of the trip recommendation. Defaults to None.
        keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.

    Returns:
        list[dict]: A list of trip recommendation dictionaries matching the search criteria.
    """
    print(f"DEBUG: search_trip_recommendations called with: location={location}, name={name}, keywords={keywords}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM trip_recommendations WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if keywords:
        keyword_list = keywords.split(",")
        keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
        query += f" AND ({keyword_conditions})"
        params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

    print(f"DEBUG: search_trip_recommendations SQL query: {query} with params: {params}") # DEBUG
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    dict_results = [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]
    print(f"DEBUG: search_trip_recommendations returning {len(dict_results)} results.") # DEBUG
    return dict_results


@tool
def book_excursion(recommendation_id: int) -> str:
    """
    Book a excursion by its recommendation ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to book.

    Returns:
        str: A message indicating whether the trip recommendation was successfully booked or not.
    """
    print(f"DEBUG: book_excursion called with recommendation_id: {recommendation_id}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        print(f"DEBUG: book_excursion returning 'Trip recommendation {recommendation_id} successfully booked.'") # DEBUG
        return f"Trip recommendation {recommendation_id} successfully booked."
    else:
        conn.close()
        print(f"DEBUG: book_excursion returning 'No trip recommendation found with ID {recommendation_id}.'") # DEBUG
        return f"No trip recommendation found with ID {recommendation_id}."


@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """
    Update a trip recommendation's details by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to update.
        details (str): The new details of the trip recommendation.

    Returns:
        str: A message indicating whether the trip recommendation was successfully updated or not.
    """
    print(f"DEBUG: update_excursion called with recommendation_id={recommendation_id}, details='{details}'") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET details = ? WHERE id = ?",
        (details, recommendation_id),
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        print(f"DEBUG: update_excursion returning 'Trip recommendation {recommendation_id} successfully updated.'") # DEBUG
        return f"Trip recommendation {recommendation_id} successfully updated."
    else:
        conn.close()
        print(f"DEBUG: update_excursion returning 'No trip recommendation found with ID {recommendation_id}.'") # DEBUG
        return f"No trip recommendation found with ID {recommendation_id}."


@tool
def cancel_excursion(recommendation_id: int) -> str:
    """
    Cancel a trip recommendation by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to cancel.

    Returns:
        str: A message indicating whether the trip recommendation was successfully cancelled or not.
    """
    print(f"DEBUG: cancel_excursion called with recommendation_id: {recommendation_id}") # DEBUG
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        print(f"DEBUG: cancel_excursion returning 'Trip recommendation {recommendation_id} successfully cancelled.'") # DEBUG
        return f"Trip recommendation {recommendation_id} successfully cancelled."
    else:
        conn.close()
        print(f"DEBUG: cancel_excursion returning 'No trip recommendation found with ID {recommendation_id}.'") # DEBUG
        return f"No trip recommendation found with ID {recommendation_id}."


# In[46]:


# Define Utilities

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    print(f"DEBUG: handle_tool_error called. Error: {repr(error)}") # DEBUG
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
            # Add a specific check for empty tool messages
            if isinstance(message, ToolMessage) and not message.content:
                print(f"DEBUG: !!! Detected empty ToolMessage content for tool_call_id: {message.tool_call_id} !!!") # DEBUG


# ### 1. pre. define (STATAE) & (GRAPH) structure

# In[47]:


from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]


# #### define each assistant, build class object 

# In[48]:


from datetime import date, datetime
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from pydantic import BaseModel, Field


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }

llm = ChatOpenAI(model="gpt-4o")
        
# Flight booking assistant

flight_booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling flight updates. "
            " The primary assistant delegates work to you whenever the user needs help updating their bookings. "
            "Confirm the updated flight details with the customer and inform them of any additional fees. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.',
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

update_flight_safe_tools = [search_flights]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools
update_flight_runnable = flight_booking_prompt | llm.bind_tools(
    update_flight_tools + [CompleteOrEscalate]
)

# Hotel Booking Assistant
book_hotel_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling hotel bookings. "
            "The primary assistant delegates work to you whenever the user needs help booking a hotel. "
            "Search for available hotels based on the user's preferences and confirm the booking details with the customer. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.'
            " Do not waste the user's time. Do not make up invalid tools or functions."
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'what's the weather like this time of year?'\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'i need to figure out transportation while i'm there'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Hotel booking confirmed'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_hotel_safe_tools = [search_hotels]
book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
book_hotel_tools = book_hotel_safe_tools + book_hotel_sensitive_tools
book_hotel_runnable = book_hotel_prompt | llm.bind_tools(
    book_hotel_tools + [CompleteOrEscalate]
)

# Car Rental Assistant
book_car_rental_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling car rental bookings. "
            "The primary assistant delegates work to you whenever the user needs help booking a car rental. "
            "Search for available car rentals based on the user's preferences and confirm the booking details with the customer. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
            '"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'what's the weather like this time of year?'\n"
            " - 'What flights are available?'\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Car rental booking confirmed'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_car_rental_safe_tools = [search_car_rentals]
book_car_rental_sensitive_tools = [
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
]
book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools
book_car_rental_runnable = book_car_rental_prompt | llm.bind_tools(
    book_car_rental_tools + [CompleteOrEscalate]
)

# Excursion Assistant

book_excursion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling trip recommendations. "
            "The primary assistant delegates work to you whenever the user needs help booking a recommended trip. "
            "Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'i need to figure out transportation while i'm there'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Excursion booking confirmed!'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_excursion_safe_tools = [search_trip_recommendations]
book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools
book_excursion_runnable = book_excursion_prompt | llm.bind_tools(
    book_excursion_tools + [CompleteOrEscalate]
)


# Primary Assistant
class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
    )


class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""

    location: str = Field(
        description="The location where the user wants to rent a car."
    )
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the car rental."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Basel",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "I need a compact car with automatic transmission.",
            }
        }


class ToHotelBookingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle hotel bookings."""

    location: str = Field(
        description="The location where the user wants to book a hotel."
    )
    checkin_date: str = Field(description="The check-in date for the hotel.")
    checkout_date: str = Field(description="The check-out date for the hotel.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the hotel booking."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Zurich",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "I prefer a hotel near the city center with a room that has a view.",
            }
        }


class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""

    location: str = Field(
        description="The location where the user wants to book a recommended trip."
    )
    request: str = Field(
        description="Any additional information or requests from the user regarding the trip recommendation."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Lucerne",
                "request": "The user is interested in outdoor activities and scenic views.",
            }
        }


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Your primary role is to search for flight information and company policies to answer customer queries. "
            "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)
primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToFlightBookingAssistant,
        ToBookCarRental,
        ToHotelBookingAssistant,
        ToBookExcursion,
    ]
)


# In[49]:


from typing import Callable

from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        print(f"DEBUG: Entering {assistant_name} with tool_call_id: {tool_call_id}") # DEBUG
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


# ### 1.2 graph structure

# In[50]:


from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.graph import START, END


builder = StateGraph(State)


def user_info(state: State):
    print("DEBUG: Executing user_info node.") # DEBUG
    return {"user_info": fetch_user_flight_information.invoke({})}


builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")


# #### define 5 nodes for graph structure by grapg_builder (StateGraph) 
# #### for EACH 5 MINI-Assistants  ( flight -> rental -> hotel -> booking excur -----> primary assistant)
# 
# enter_*
# 
# 
# Assistant
# 
# 
# *_safe_tools
# 
# 
# *_sensitive_tools
# 
# 
# leave_skill

# #### 1st mini assistant, 'flight'

# In[51]:


# Flight booking assistant

# enter_
builder.add_node(
    "enter_update_flight",
    create_entry_node("Flight Updates & Booking Assistant", "update_flight"),
)
# assist_
builder.add_node("update_flight", Assistant(update_flight_runnable))
builder.add_edge("enter_update_flight", "update_flight")
# safe_tool
builder.add_node(
    "update_flight_sensitive_tools",
    create_tool_node_with_fallback(update_flight_sensitive_tools),
)
# sensi_tool
builder.add_node(
    "update_flight_safe_tools",
    create_tool_node_with_fallback(update_flight_safe_tools),
)


def route_update_flight(
    state: State,
):
    print(f"DEBUG: Routing in update_flight. Current state messages: {state['messages']}") # DEBUG
    route = tools_condition(state)
    print(f"DEBUG: tools_condition route for update_flight: {route}") # DEBUG
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        print("DEBUG: Route update_flight to leave_skill (CompleteOrEscalate).") # DEBUG
        return "leave_skill"
    safe_toolnames = [t.name for t in update_flight_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        print("DEBUG: Route update_flight to update_flight_safe_tools.") # DEBUG
        return "update_flight_safe_tools"
    print("DEBUG: Route update_flight to update_flight_sensitive_tools.") # DEBUG
    return "update_flight_sensitive_tools"


builder.add_edge("update_flight_sensitive_tools", "update_flight")
builder.add_edge("update_flight_safe_tools", "update_flight")
# conditional, leave_skill
builder.add_conditional_edges(
    "update_flight",
    route_update_flight,
    ["update_flight_sensitive_tools", "update_flight_safe_tools", "leave_skill", END],
)


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    print("DEBUG: Popping dialog state.") # DEBUG
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")


# #### 2nd mini assistant, 'rental'

# In[52]:


# Car rental assistant


# enter_
builder.add_node(
    "enter_book_car_rental",
    create_entry_node("Car Rental Assistant", "book_car_rental"),
)
# tools
builder.add_node("book_car_rental", Assistant(book_car_rental_runnable))
builder.add_edge("enter_book_car_rental", "book_car_rental")

builder.add_node(
    "book_car_rental_safe_tools",
    create_tool_node_with_fallback(book_car_rental_safe_tools),
)
builder.add_node(
    "book_car_rental_sensitive_tools",
    create_tool_node_with_fallback(book_car_rental_sensitive_tools),
)


def route_book_car_rental(
    state: State,
):
    print(f"DEBUG: Routing in book_car_rental. Current state messages: {state['messages']}") # DEBUG
    route = tools_condition(state)
    print(f"DEBUG: tools_condition route for book_car_rental: {route}") # DEBUG
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        print("DEBUG: Route book_car_rental to leave_skill (CompleteOrEscalate).") # DEBUG
        return "leave_skill"
    safe_toolnames = [t.name for t in book_car_rental_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        print("DEBUG: Route book_car_rental to book_car_rental_safe_tools.") # DEBUG
        return "book_car_rental_safe_tools"
    print("DEBUG: Route book_car_rental to book_car_rental_sensitive_tools.") # DEBUG
    return "book_car_rental_sensitive_tools"


builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
builder.add_edge("book_car_rental_safe_tools", "book_car_rental")
builder.add_conditional_edges(
    "book_car_rental",
    route_book_car_rental,
    [
        "book_car_rental_safe_tools",
        "book_car_rental_sensitive_tools",
        "leave_skill",
        END,
    ],
)


# #### 3rd mini assistant, 'hotel' booking

# In[53]:


# Hotel booking assistant
builder.add_node(
    "enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel")
)
builder.add_node("book_hotel", Assistant(book_hotel_runnable))
builder.add_edge("enter_book_hotel", "book_hotel")
builder.add_node(
    "book_hotel_safe_tools",
    create_tool_node_with_fallback(book_hotel_safe_tools),
)
builder.add_node(
    "book_hotel_sensitive_tools",
    create_tool_node_with_fallback(book_hotel_sensitive_tools),
)


def route_book_hotel(
    state: State,
):
    print(f"DEBUG: Routing in book_hotel. Current state messages: {state['messages']}") # DEBUG
    route = tools_condition(state)
    print(f"DEBUG: tools_condition route for book_hotel: {route}") # DEBUG
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        print("DEBUG: Route book_hotel to leave_skill (CompleteOrEscalate).") # DEBUG
        return "leave_skill"
    tool_names = [t.name for t in book_hotel_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        print("DEBUG: Route book_hotel to book_hotel_safe_tools.") # DEBUG
        return "book_hotel_safe_tools"
    print("DEBUG: Route book_hotel to book_hotel_sensitive_tools.") # DEBUG
    return "book_hotel_sensitive_tools"


builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
builder.add_edge("book_hotel_safe_tools", "book_hotel")
builder.add_conditional_edges(
    "book_hotel",
    route_book_hotel,
    ["leave_skill", "book_hotel_safe_tools", "book_hotel_sensitive_tools", END],
)


# #### 4th mini assistant, 'excursion'

# In[54]:


# Excursion assistant
builder.add_node(
    "enter_book_excursion",
    create_entry_node("Trip Recommendation Assistant", "book_excursion"),
)
builder.add_node("book_excursion", Assistant(book_excursion_runnable))
builder.add_edge("enter_book_excursion", "book_excursion")
builder.add_node(
    "book_excursion_safe_tools",
    create_tool_node_with_fallback(book_excursion_safe_tools),
)
builder.add_node(
    "book_excursion_sensitive_tools",
    create_tool_node_with_fallback(book_excursion_sensitive_tools),
)


def route_book_excursion(
    state: State,
):
    print(f"DEBUG: Routing in book_excursion. Current state messages: {state['messages']}") # DEBUG
    route = tools_condition(state)
    print(f"DEBUG: tools_condition route for book_excursion: {route}") # DEBUG
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        print("DEBUG: Route book_excursion to leave_skill (CompleteOrEscalate).") # DEBUG
        return "leave_skill"
    tool_names = [t.name for t in book_excursion_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        print("DEBUG: Route book_excursion to book_excursion_safe_tools.") # DEBUG
        return "book_excursion_safe_tools"
    print("DEBUG: Route book_excursion to book_excursion_sensitive_tools.") # DEBUG
    return "book_excursion_sensitive_tools"


builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
builder.add_edge("book_excursion_safe_tools", "book_excursion")
builder.add_conditional_edges(
    "book_excursion",
    route_book_excursion,
    ["book_excursion_safe_tools", "book_excursion_sensitive_tools", "leave_skill", END],
)


# In[55]:


# ----> Primary assistant

builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
)


def route_primary_assistant(
    state: State,
):
    print(f"DEBUG: Routing in primary_assistant. Current state messages: {state['messages']}") # DEBUG
    route = tools_condition(state)
    print(f"DEBUG: tools_condition route for primary_assistant: {route}") # DEBUG
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            print("DEBUG: Route primary_assistant to enter_update_flight.") # DEBUG
            return "enter_update_flight"
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            print("DEBUG: Route primary_assistant to enter_book_car_rental.") # DEBUG
            return "enter_book_car_rental"
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            print("DEBUG: Route primary_assistant to enter_book_hotel.") # DEBUG
            return "enter_book_hotel"
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            print("DEBUG: Route primary_assistant to enter_book_excursion.") # DEBUG
            return "enter_book_excursion"
        print("DEBUG: Route primary_assistant to primary_assistant_tools.") # DEBUG
        return "primary_assistant_tools"
    print("DEBUG: Route primary_assistant to END (no tool calls).") # DEBUG
    raise ValueError("Invalid route")


# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_update_flight",
        "enter_book_car_rental",
        "enter_book_hotel",
        "enter_book_excursion",
        "primary_assistant_tools",
        END,
    ],
)
builder.add_edge("primary_assistant_tools", "primary_assistant")


# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "update_flight",
    "book_car_rental",
    "book_hotel",
    "book_excursion",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    print(f"DEBUG: Routing to workflow. Current dialog_state: {dialog_state}") # DEBUG
    if not dialog_state:
        print("DEBUG: Routing to primary_assistant (no dialog state).") # DEBUG
        return "primary_assistant"
    print(f"DEBUG: Routing to {dialog_state[-1]}.") # DEBUG
    return dialog_state[-1]


builder.add_conditional_edges("fetch_user_info", route_to_workflow)

# After DONE adding node/edge, compile()
# Compile graph
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ],
)


# In[59]:


graph


# ### Finished (State+Graph)
# ### 2 Conversation & start using the bot

# In[170]:


import shutil
import uuid

# Update with the backup file so we can restart from the original place in each section
db = update_dates(db)
thread_id = str( uuid.uuid4() )

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3443 587242", # Ensure this passenger_id exists in your DB
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

_printed = set()

questions = ["I need some expert guidance travel planning and iternarary guide. Could you request assistance for me?",
            
            " It's much more reliable and extensible to have some help during planning."
             
            ]

print("\n--- Starting graph.stream() loop for initial questions ---") # DEBUG
for question in questions:
    print(f"\nDEBUG: Streaming for user question: '{question}'") # DEBUG
    events = graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )

    for event in events:
        _print_event(event, _printed)
    snapshot = graph.get_state(config)
    while snapshot.next:
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
        try:
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
        except:
            user_input = "y"
        if user_input.strip() == "y":
            # Just continue
            result = graph.invoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        snapshot = graph.get_state(config)
print("\n--- Finished graph.stream() loop ---") # DEBUG


# ### 3 Eval
# ### load the bot to eval file and start performance review

# In[171]:


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

system_prompt_template = """You are a customer of an airline company. \
You are interacting with a user who is a customer support person. \

{instructions}

When you are finished with the conversation, respond with a single word 'FINISHED'"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
instructions = """Your name is Jason. You are trying to get a refund for the trip you took to Alaska. \
You want them to give you ALL the money back. \
This trip happened 5 years ago."""

prompt = prompt.partial(name="JJ", instructions=instructions)

model = ChatOpenAI()

simulated_user = prompt | model


# In[172]:


from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Hi! I would like information regarding travel planning")]
simulated_user.invoke({"messages": messages})


# ### Get Passenger_id and pass the same id into my_chat_bot func for evaluation

# In[173]:


config.get("configurable", {}).get("passenger_id", None)


# In[174]:


from typing import List

def my_chat_bot(messages: List[dict]) -> dict:
    """
    Args:
        messages_openai_format: A list of messages in OpenAI format
                                (e.g., [{"role": "user", "content": "hello"}]).

    Returns:
        A dictionary with the chatbot's response content, e.g., {"content": "..."}.
    """
    global thread_id, graph # Access the global thread ID and graph

    print(f"\n--- my_chat_bot received messages: {messages}") # DEBUG

    # 1. Convert OpenAI format messages to LangChain BaseMessage objects
    langchain_messages: List[BaseMessage] = [] # FIX: Initialize as empty list
    for m in messages:
        if m["role"] == "user":
            langchain_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            langchain_messages.append(AIMessage(content=m["content"]))
        elif m["role"] == "system":
            langchain_messages.append(SystemMessage(content=m["content"]))
        # Add more roles if your graph handles them, e.g., tool messages
        # For simplicity, we are not directly parsing tool_calls from OpenAI format
        # back to LangChain ToolMessage objects here, as the mock LLM doesn't use tools.

    print(f"--- Converted LangChain messages for graph.invoke: {langchain_messages}") # DEBUG

    # 2. Define the configuration for LangGraph invocation
    # The 'thread_id' here ties the invocation to a specific conversation's state.
    # We use a single global thread_id for this example.
    config = {
        "configurable": {
            "thread_id": thread_id,
            "passenger_id" : '3443 587242', # Ensure this matches the ID used in the main conversation loop
            # You can add other configurable parameters here if your graph needs them,
            # like "passenger_id" from your original example.
            # "passenger_id": "3442 587242",
        }
    }

    print(f"--- LangGraph config for my_chat_bot: {config}") # DEBUG

    # 3. Invoke the LangGraph with the full conversation history
    # The graph will process these messages and return the next state.
    try:
        # We invoke with the entire history. LangGraph will load the state for the
        # thread_id and determine the next action based on the input messages and its internal logic.
        # The result here will be the final state of the graph.
        final_state = graph.invoke(
            {"messages": langchain_messages},
            config,
        )

        print(f"--- LangGraph final state from my_chat_bot: {final_state}") # DEBUG

        # 4. Extract the content of the last AIMessage from the graph's output
        # The 'messages' key in the final state should contain the updated message list.
        # We look for the last AIMessage.
        ai_response_content = "An error occurred or no AI response found."
        if final_state and "messages" in final_state:
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage):
                    ai_response_content = msg.content
                    break
        
        print(f"--- Extracted AI response content from my_chat_bot: {ai_response_content}") # DEBUG
        return {"content": ai_response_content}

    except Exception as e:
        print(f"Error during LangGraph invocation in my_chat_bot: {e}") # DEBUG
        # Return a fallback response in case of an error
        return {"content": f"Sorry, I encountered an error: {e}"}


# In[175]:


from langchain_community.adapters.openai import convert_message_to_dict
from langchain_core.messages import AIMessage


def chat_bot_node(state):
    messages = state["messages"]
    # Convert from LangChain format to the OpenAI format, which our chatbot function expects.
    messages = [convert_message_to_dict(m) for m in messages]
    # Call the chat bot
    chat_bot_response = my_chat_bot(messages)
    # Respond with an AI Message
    return {"messages": [AIMessage(content=chat_bot_response["content"])]}


# In[176]:


def _swap_roles(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages


def simulated_user_node(state):
    messages = state["messages"]
    # Swap roles of messages
    new_messages = _swap_roles(messages)
    # Call the simulated user
    response = simulated_user.invoke({"messages": new_messages})
    # This response is an AI message - we need to flip this to be a human message
    return {"messages": [HumanMessage(content=response.content)]}


# In[177]:


def should_continue(state):
    messages = state["messages"]
    if len(messages) > 6:
        return "end"
    elif messages[-1].content == "FINISHED":
        return "end"
    else:
        return "continue"


# In[178]:


from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
graph_builder.add_node("user", simulated_user_node)
graph_builder.add_node("chat_bot", chat_bot_node)
# Every response from  your chat bot will automatically go to the
# simulated user
graph_builder.add_edge("chat_bot", "user")
graph_builder.add_conditional_edges(
    "user",
    should_continue,
    # If the finish criteria are met, we will stop the simulation,
    # otherwise, the virtual user's message will be sent to your chat bot
    {
        "end": END,
        "continue": "chat_bot",
    },
)
# The input will first go to your chat bot
graph_builder.add_edge(START, "chat_bot")
simulation = graph_builder.compile()
simulation


# In[ ]:





# In[179]:


print("\n--- Starting simulation.stream() ---") # DEBUG
for chunk in simulation.stream({"messages": []}):
    # Print out all events aside from the final end chunk
    if END not in chunk:
        print(chunk)
        print("----")
print("\n--- Finished simulation.stream() ---") # DEBUG


# In[181]:


# Simulate the `state` object that `chat_bot_node` receives
# Initialize with an empty message list for the start of a conversation
initial_state = {"messages": []}

# Define the questions for the simulated conversation
simulated_questions = questions

print("\n--- Starting Simulated Conversation ---")



# First turn: User asks a question
user_question_1 = simulated_questions[0]
print(f"\nSimulated User Input: {user_question_1}")
initial_state["messages"].append(HumanMessage(content=user_question_1))

# Run through the chat_bot_node
result_state_1 = chat_bot_node(initial_state)
ai_response_1 = result_state_1["messages"][0]
print(f"Chatbot AI Response 1: {ai_response_1.content}")

# Add AI's response to the conversation history for the next turn
initial_state["messages"].append(ai_response_1)



# Second turn: Simulate a follow-up interaction (if your graph handles it as a user input)
# Note: The original `simulated_questions` list contains AI responses.
# For a true "sim user", these would be *expected* outputs or part of the internal
# script for the sim user. Let's make the second "question" a user follow-up.
user_question_2 = "Why do you say it's more reliable?"
print(f"\nSimulated User Input: {user_question_2}")
initial_state["messages"].append(HumanMessage(content=user_question_2))

result_state_2 = chat_bot_node(initial_state)
ai_response_2 = result_state_2["messages"][0]
print(f"Chatbot AI Response 2: {ai_response_2.content}")

# Add AI's response to the conversation history
initial_state["messages"].append(ai_response_2)



# Third turn: Another user follow-up
user_question_3 = "Can you explain extensibility further?"
print(f"\nSimulated User Input: {user_question_3}")
initial_state["messages"].append(HumanMessage(content=user_question_3))

result_state_3 = chat_bot_node(initial_state)
ai_response_3 = result_state_3["messages"][0]
print(f"Chatbot AI Response 3: {ai_response_3.content}")

print("\n--- End Simulated Conversation ---")

