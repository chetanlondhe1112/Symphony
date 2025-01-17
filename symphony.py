# Symphony : RAG Application service 
# Version : 0.0.0

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
import os
from fastapi import Form, HTTPException, APIRouter, FastAPI
from pydantic import BaseModel
import asyncio
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
import toml

app = FastAPI(title="Symphony : Retrieval-Augmented Generation Application Service",
    description="""
    Background : 
    Symphony is RAG application for all structured and unstructured data""",
    version="0.0.0",
    contact={
        "name": "Support Team",
        "email": "chetanlondhe1112@gmail.com",
    })

script_dir = os.getcwd()
config_path = os.path.join(script_dir, "conf/conf.toml")

config = toml.load(config_path)

log_path = os.path.join(script_dir, "log/symphony_log.log")
logger.add(log_path)

router = APIRouter()

# Setting up the SQL Database Connection
logger.info("Loading Database")
db_config=config['database']['cred']
db = SQLDatabase.from_uri(f"mysql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
logger.info("Done")

gemini_api_key = config['llm']['llm_api_key']

# Configuring the OpenAI Language Model
logger.info("Model Loading")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,api_key=gemini_api_key)
logger.info("Done")

# Creating SQL Database Toolkit
logger.info("Tolkit Loading")
toolkit = SQLDatabaseToolkit(db=db, llm=model)
logger.info("Done")

# Creating and Running a SQL Agent
logger.info("SQL Agent Loading")
agent_executor = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
logger.info("Done")

@app.post("/")
async def arnoldtest_rager(query : str = Form(...)):
    logger.info(f"Query :{query}")
    try:
        result = agent_executor.invoke(str(query),return_only_outputs=True,include_run_info=False)
        logger.info(result['output'])
        return {"result":result['output']}
    except Exception as e:
        return {"result": e}
