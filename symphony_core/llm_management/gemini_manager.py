from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio

class Geminihandler:
    
    def __init__(self,llm_config):
        self.llm_config = llm_config

        self.llm_model_name = self.llm_config['llm_name']
        self.llm_model = self.llm_config['llm_model']
        self.model_api_key = self.llm_config['llm_api_key']
        self.model_temperature = self.llm_config['llm_temperature']
        self.model_embeddings = self.llm_config['llm_embeddings_model']
        
    async def get_model(self)->str:
        return await ChatGoogleGenerativeAI(model=self.llm_model, temperature=self.model_temperature,api_key=self.model_api_key)