# Symphony : RAG Application service 
# Version : 0.0.0

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
import os
from fastapi import Form, HTTPException, APIRouter
from pydantic import BaseModel
import asyncio
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(title="AIRBOT : AI Response BOT Service API's",
    description="""
    Background : 
    Symphony is RAG application for all structured and unstructured data""",
    version="0.0.0",
    contact={
        "name": "Support Team",
        "email": "chetanlondhe1112@gmail.com",
    })

script_dir = os.getcwd()
log_path = os.path.join(script_dir, "log\\symphony_log.log")
logger.add(log_path)

router = APIRouter()

# Setting up the SQL Database Connection
logger.info("Loading Database")
db = SQLDatabase.from_uri("mysql://root:@localhost:3306/arnoldtest")
logger.info("Done")

gemini_api_key = "AIzaSyBXtONwBU3CE8sXud76m87iXnaQRq18HHw"

# Configuring the OpenAI Language Model
logger.info("Model Loading")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,api_key=gemini_api_key)
logger.info("Done")

# Creating SQL Database Toolkit
logger.info("Model Loading")
toolkit = SQLDatabaseToolkit(db=db, llm=model)
logger.info("Done")

# Creating and Running a SQL Agent
logger.info("Done")
agent_executor = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
logger.info("Done")

@app.post("/arnoldtest_rager/")
async def textmaster(query : str = Form(...)):
    logger.info("Query :",query)
    result = await agent_executor.invoke(str(query),return_only_outputs=True,include_run_info=False)
    logger.info(result['output'])
    return {"result":result['output']}
