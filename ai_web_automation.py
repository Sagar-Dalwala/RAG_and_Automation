from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
from typing import Dict, List, Union, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
from web_automation import WebAutomation as BaseWebAutomation

class WebAction(BaseModel):
    action_type: str = Field(..., description="Type of web action to perform")
    parameters: Dict = Field(default_factory=dict, description="Parameters for the action")

class WebTaskPlan(BaseModel):
    steps: List[WebAction] = Field(..., description="List of web actions to perform")

class WebAutomation(BaseWebAutomation):
    def __init__(self):
        super().__init__()

    async def navigate_to_url(self, url):
        result = super().navigate_to_url(url)
        return result

    async def extract_page_content(self):
        result = super().extract_page_content()
        return result

    async def search_in_page(self, query):
        result = super().search_in_page(query)
        return result

    async def scroll_page(self, direction="down"):
        result = super().scroll_page(direction)
        return result

    async def take_screenshot(self, filename):
        result = super().take_screenshot(filename)
        return result

class AIWebAutomation:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.web_automation = WebAutomation()
        self.setup_task_planning()

    def setup_task_planning(self):
        """Setup the task planning system"""
        self.task_planning_prompt = ChatPromptTemplate.from_template(
            """Given the following user command, create a detailed plan of web automation steps.
            Break down complex tasks into simple, executable actions.
            
            User Command: {command}
            
            Create a plan with a list of steps. Each step should have an action_type and parameters.
            Valid action types: navigate, extract_content, search, click, input_text, scroll, wait
            
            For wait actions, use 'selector' and 'timeout' parameters.
            For click and input actions, use 'selector' and 'selector_type' parameters.
            For scroll actions, use 'direction' and 'amount' parameters.
            
            Respond with a valid JSON object containing a 'steps' array of actions.
            Each action must have 'action_type' and 'parameters' fields.
            Do not include any explanatory text or markdown formatting.
            
            Example response format:
            {{
                "steps": [
                    {{
                        "action_type": "navigate",
                        "parameters": {{
                            "url": "example.com"
                        }}
                    }},
                    {{
                        "action_type": "wait",
                        "parameters": {{
                            "selector": "#main-content",
                            "timeout": 10
                        }}
                    }}
                ]
            }}
            """
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=WebTaskPlan)

    async def execute_command(self, command: str) -> Dict:
        """Execute a natural language command for web automation"""
        try:
            # Generate task plan
            task_plan = await self._generate_task_plan(command)
            
            # Execute each step in the plan
            results = []
            for step in task_plan.steps:
                result = await self._execute_step(step)
                results.append(result)
                
                if "error" in result:
                    break
                    
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"error": f"Failed to execute command: {str(e)}"}

    async def _generate_task_plan(self, command: str) -> WebTaskPlan:
        """Generate a task plan from a natural language command"""
        chain = self.task_planning_prompt | self.llm | self.output_parser
        return await chain.ainvoke({"command": command})

    async def _execute_step(self, step: WebAction) -> Dict:
        """Execute a single step in the task plan"""
        action_map = {
            "navigate": self.web_automation.navigate_to_url,
            "extract_content": self.web_automation.extract_page_content,
            "search": self.web_automation.search_in_page,
            "click": self._click_element,
            "input_text": self._input_text,
            "scroll": self._scroll_page,
            "wait": self._wait_for_element
        }
        
        if step.action_type not in action_map:
            return {"error": f"Unknown action type: {step.action_type}"}
            
        try:
            action_func = action_map[step.action_type]
            result = await action_func(**step.parameters)
            return {"success": True, "action": step.action_type, "result": result}
        except Exception as e:
            return {"error": f"Failed to execute {step.action_type}: {str(e)}"}

    async def _click_element(self, selector: str, selector_type: str = "css") -> Dict:
        """Click an element on the page"""
        try:
            by_map = {"css": By.CSS_SELECTOR, "xpath": By.XPATH, "id": By.ID}
            by_type = by_map.get(selector_type, By.CSS_SELECTOR)
            
            element = WebDriverWait(self.web_automation.driver, 10).until(
                EC.element_to_be_clickable((by_type, selector))
            )
            element.click()
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    async def _input_text(self, selector: str, text: str, selector_type: str = "css") -> Dict:
        """Input text into a form field"""
        try:
            by_map = {"css": By.CSS_SELECTOR, "xpath": By.XPATH, "id": By.ID}
            by_type = by_map.get(selector_type, By.CSS_SELECTOR)
            
            element = WebDriverWait(self.web_automation.driver, 10).until(
                EC.presence_of_element_located((by_type, selector))
            )
            element.clear()
            element.send_keys(text)
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    async def _scroll_page(self, direction: str = "down", amount: int = 300) -> Dict:
        """Scroll the page"""
        try:
            scroll_script = f"window.scrollBy(0, {amount if direction == 'down' else -amount})"
            self.web_automation.driver.execute_script(scroll_script)
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    async def _wait_for_element(self, selector: str, selector_type: str = "css", timeout: int = 10) -> Dict:
        """Wait for an element to appear on the page"""
        try:
            by_map = {"css": By.CSS_SELECTOR, "xpath": By.XPATH, "id": By.ID}
            by_type = by_map.get(selector_type, By.CSS_SELECTOR)
            
            WebDriverWait(self.web_automation.driver, timeout).until(
                EC.presence_of_element_located((by_type, selector))
            )
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}