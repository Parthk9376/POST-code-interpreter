import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# CORS is required for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact stdout or traceback.
    """

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {"__name__": "__main__"})

        output = sys.stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """
    Use Gemini structured output to identify the exact error line.
    """

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return []

    client = genai.Client(api_key=api_key)

    prompt = f"""
Analyze the following Python code and its error traceback.

Identify the exact source code line number or line numbers
where the error occurred.

Return only the line numbers where the error is located.

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.INTEGER
                        )
                    )
                },
                required=["error_lines"]
            )
        )
    )

    result = ErrorAnalysis.model_validate_json(response.text)

    return result.error_lines


@app.get("/")
def root():
    return {
        "message": "Code Interpreter API is running"
    }


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # Successful execution
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Error occurred: ask AI for exact line numbers
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }
