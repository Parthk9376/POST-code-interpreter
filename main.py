import os
import sys
import re
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types


app = FastAPI()

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


def get_line_from_traceback(traceback_text: str) -> List[int]:
    """
    Extract Python source-code line numbers from a traceback.
    """

    lines = re.findall(
        r'File ".*?", line (\d+)',
        traceback_text
    )

    if not lines:
        return []

    return [int(lines[-1])]


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:

    api_key = os.environ.get("GEMINI_API_KEY")

    # Try AI analysis first
    if api_key:

        try:
            client = genai.Client(api_key=api_key)

            prompt = f"""
Analyze this Python code and traceback.

Identify the exact source-code line number or line numbers
where the error occurred.

Return the line numbers in the required JSON structure.

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

            result = ErrorAnalysis.model_validate_json(
                response.text
            )

            if result.error_lines:
                return result.error_lines

        except Exception:
            pass

    # Reliable fallback using the exact Python traceback.
    return get_line_from_traceback(traceback_text)


@app.get("/")
def root():
    return {
        "message": "Code Interpreter API is running"
    }


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # No error
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Error occurred
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }
