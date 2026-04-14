import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tools.data_tools import analyze_sensor_trends, check_maintenance_sensors, check_market_prices, consult_technical_manual, get_failed_machines, get_commodity_price, get_supplier_info, run_sql_query, get_market_news, predict_failure
from tools.search_tools import get_live_market_news
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory



load_dotenv()

import logging
import os

# 1. Create the logs folder automatically if it doesn't exist
os.makedirs('logs', exist_ok=True)

# 2. Professional Log Configuration
logging.basicConfig(
    filename='logs/factory_brain.log',
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("VULCAN")

# Inside your IndustrialAI class __init__, you can log that it started:
# logger.info("VULCAN AI Brain initialized successfully.")

class IndustrialAI:
    def __init__(self):
        try:
            # Existing initialization...
            logger.info("Industrial AI Brain initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize AI Brain: {e}")
            raise
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        # 1. Define the Tools available to the AI
        self.tools = [analyze_sensor_trends, check_maintenance_sensors, check_market_prices, consult_technical_manual, run_sql_query, get_market_news, predict_failure]

        # 2. Define a "Smart" Prompt with Memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Senior Industrial Systems Engineer. You have autonomous access to sensor data, repair manuals, and market prices. Use tools only when necessary to provide factual answers."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 3. Create the Agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        # 4. Create the Executor (The loop that runs the agent)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, memory=self.memory, verbose=True)

        self.system_prompt = """
        You are a Senior Industrial Systems Engineer.
        If a technical manual gives qualitative advice (like 'excessive torque'), use the 'analyze_sensor_trends' tool 
        to see if the current value is statistically higher than the factory average.
        Always link Technical Risks (Manual) with Statistical Reality (Data) and Financial Impact (Market Prices).
        """

    def run_analysis(self):
        print("🤖 Step 1: Analyzing Maintenance Logs...")
        failures = get_failed_machines()
        
        if not failures:
            return "All systems nominal. No failures detected."

        # We take the first critical failure to analyze
        target = failures[0]
        fail_type = target['failure_type']
        
        print(f"⚠️ Failure Detected: {fail_type} on Machine {target['product_id']}")

        # 2. Market Research (Mapping Failure to Material)
        # We assume: TWF/OSF = Steel, PWF = Copper, HDF = Aluminum
        material_map = {"TWF": "Steel", "OSF": "Iron Ore", "PWF": "Copper", "HDF": "Aluminum"}
        material = material_map.get(fail_type, "Steel")
        
        print(f"🔍 Step 2: Checking Market Prices for {material}...")
        market_data = get_commodity_price(material)
        
        print("🚛 Step 3: Checking Supplier Availability...")
        supplier_data = get_supplier_info()

        print("🌐 Step 4: Fetching Live Market News...")
        search_query = f"current {material} market supply chain disruptions March 2026"
        live_news = get_live_market_news(search_query)

        # 4. Final Executive Strategy
        print("📊 Step 5: Generating Executive Report...")
        prompt = f"""
        CONTEXT:
        - Machine Failure: {fail_type}
        - Material: {material}
        - Latest Market Price: {market_data}
        - Best Suppliers: {supplier_data}
        - LIVE WORLD NEWS: {live_news}
        
        TASK:
        Write a 3-paragraph Executive Report. 
        Para 1: The technical risk of this failure.
        Para 2: The financial impact based on current commodity prices AND mention if LIVE WORLD NEWS 
        (e.g., strikes, supply chain disruptions) increases urgency.
        Para 3: Action plan (Which supplier to use and why).
        """
        response = self.llm.invoke([SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)])
        return response.content

if __name__ == "__main__":
    agent = IndustrialAI()
    report = agent.run_analysis()
    print("\n--- FINAL REPORT ---\n")
    print(report)