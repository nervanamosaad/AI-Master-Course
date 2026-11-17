import ast
import re
import sqlite3
import sys
import requests

from pathlib import Path


# CONFIGURATION

MODEL = "gemma2:2b"
OLLAMA_URL = "http://localhost:11434/api/chat"

MAX_STEPS = 5
MAX_ROWS = 20

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sample.db"
WORKSPACE = BASE_DIR / "workspace"


# CREATE SAMPLE DATABASE AND FILES
def create_sample_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            id INTEGER PRIMARY KEY,
            name TEXT,
            governorate TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY,
            station_id INTEGER,
            month TEXT,
            salinity_ppm REAL,
            water_level REAL,
            FOREIGN KEY (station_id) REFERENCES stations(id)
        )
    """)

    count = cursor.execute(
        "SELECT COUNT(*) FROM stations"
    ).fetchone()[0]

    if count == 0:
        stations = [
            (1, "Aswan Gauge", "Aswan"),
            (2, "Qena Canal", "Qena"),
            (3, "Cairo Intake", "Cairo"),
            (4, "Giza Branch", "Giza"),
            (5, "Fayoum Drain 1", "Fayoum"),
            (6, "Alexandria Canal", "Alexandria")
        ]

        cursor.executemany(
            "INSERT INTO stations VALUES (?, ?, ?)",
            stations
        )
        measurements = [
            (1, 1, "2026-01", 250, 5.1),
            (2, 1, "2026-02", 260, 5.2),
            (3, 1, "2026-03", 270, 5.3),

            (4, 2, "2026-01", 400, 4.8),
            (5, 2, "2026-02", 420, 4.9),
            (6, 2, "2026-03", 410, 5.0),

            (7, 3, "2026-01", 300, 5.0),
            (8, 3, "2026-02", 320, 5.1),
            (9, 3, "2026-03", 310, 5.2),

            (10, 4, "2026-01", 500, 4.5),
            (11, 4, "2026-02", 520, 4.6),
            (12, 4, "2026-03", 510, 4.7),

            (13, 5, "2026-01", 2900, 3.8),
            (14, 5, "2026-02", 3000, 3.9),
            (15, 5, "2026-03", 3100, 4.0),

            (16, 6, "2026-01", 600, 4.2),
            (17, 6, "2026-02", 620, 4.3),
            (18, 6, "2026-03", 610, 4.4)
        ]

        cursor.executemany(
            "INSERT INTO measurements VALUES (?, ?, ?, ?, ?)",
            measurements
        )

    conn.commit()
    conn.close()

def create_sample_files():
    WORKSPACE.mkdir(exist_ok=True)

    files = {
        "reports/aswan_report.txt":
            "Aswan water monitoring report.\n"
            "Water level measurements are stable.\n",

        "reports/fayoum_drain_report.txt":
            "Fayoum Drain 1 report.\n"
            "Salinity reached 3100 ppm in March.\n",

        "notes/meeting_notes.txt":
            "Meeting notes about water monitoring.\n"
            "Budget for new sensors: 45000 EGP.\n",

        "data/sampling_schedule.csv":
            "station,month,status\n"
            "Aswan,March,completed\n"
            "Fayoum,March,completed\n"
    }

    for filename, content in files.items():

        file_path = WORKSPACE / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.write_text(
                content,
                encoding="utf-8"
            )

def setup_environment():
    create_sample_db()
    create_sample_files()


# TOOL1: SAFE SQLITE QUERY
def sqlite_query(query):
    query = query.strip()

    if not re.match(r"^SELECT\b", query, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    if ";" in query.rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")

    conn = None

    try:
        db_uri = DB_PATH.resolve().as_uri() + "?mode=ro"        # read only
        conn = sqlite3.connect(
            db_uri,
            uri=True,
            timeout=5
        )

        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchmany(MAX_ROWS)

        if not rows:
            return "Query executed successfully. No rows found."

        columns = rows[0].keys()
        output = [" | ".join(columns)]

        for row in rows:
            output.append(
                " | ".join(str(value) for value in row)
            )

        return "\n".join(output)

    finally:
        if conn:
            conn.close()


# TOOL2: SAFE PYTHON CALCULATOR
def eval_node(node):
    if isinstance(node, ast.Expression):
        return eval_node(node.body)

    if isinstance(node, ast.Constant):

        if type(node.value) not in (int, float):
            raise ValueError("Only numbers are allowed.")

        return node.value

    if isinstance(node, ast.BinOp):
        left = eval_node(node.left)
        right = eval_node(node.right)

        if isinstance(node.op, ast.Add):
            result = left + right

        elif isinstance(node.op, ast.Sub):
            result = left - right

        elif isinstance(node.op, ast.Mult):
            result = left * right

        elif isinstance(node.op, ast.Div):
            result = left / right

        elif isinstance(node.op, ast.FloorDiv):
            result = left // right

        elif isinstance(node.op, ast.Mod):
            result = left % right

        elif isinstance(node.op, ast.Pow):

            if abs(right) > 10:
                raise ValueError("Exponent is too large.")

            result = left ** right

        else:
            raise ValueError("Unsupported mathematical operation.")

        if abs(result) > 1e12:
            raise ValueError("Result is too large.")

        return result

    if isinstance(node, ast.UnaryOp):
        value = eval_node(node.operand)

        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value

    raise ValueError("This expression is not allowed.")


def calculator(expression):
    tree = ast.parse(expression, mode="eval")
    result = eval_node(tree)
    return str(result)


# TOOL3: LOCAL FILE SEARCH
def file_search(query):
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    results = []
    allowed_extensions = {
        ".txt", ".md", ".csv", ".log"
    }

    for file_path in WORKSPACE.rglob("*"):          # Recursive Search
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in allowed_extensions:
            continue

        try:
            resolved_path = file_path.resolve()

            # Prevent access outside workspace
            resolved_path.relative_to(WORKSPACE.resolve())

            relative_path = file_path.relative_to(WORKSPACE)

            if query.lower() in file_path.name.lower():

                results.append(
                    f"[filename] {relative_path}"
                )

            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            for line_number, line in enumerate(
                content.splitlines(), start=1
            ):

                if query.lower() in line.lower():

                    results.append(
                        f"[text] {relative_path}:{line_number}: "
                        f"{line.strip()}"
                    )

        except (OSError, ValueError):
            continue

    if not results:
        return "No matching files found."

    return "\n".join(results[:20])


# TOOL DISPATCHER
def run_tool(tool_name, tool_input):
    try:
        if not isinstance(tool_input, str) or not tool_input.strip():
            raise ValueError("Tool input cannot be empty.")

        if tool_name == "sqlite_query":
            return sqlite_query(tool_input)

        elif tool_name == "calculator":
            return calculator(tool_input)

        elif tool_name == "file_search":
            return file_search(tool_input)

        else:
            raise ValueError(
                f"Unknown tool '{tool_name}'. "
                "Valid tools: sqlite_query, calculator, file_search."
            )

    except Exception as error:
        return f"ERROR: {type(error).__name__}: {error}"


# SYSTEM PROMPT

SYSTEM_PROMPT = """
You are a ReAct agent running locally.

You have three tools:
1. sqlite_query
   Search the SQLite database.
   Tables:
   stations(id, name, governorate)
   measurements(id, station_id, month, salinity_ppm, water_level)

IMPORTANT DATABASE RULE:
The month column stores the complete date as YYYY-MM.
For example, March 2026 is stored as '2026-03'.

There is NO separate year column.
Always use the existing column names.

2. calculator
   Perform safe mathematical calculations.

Calculator input rules:
- Send only the mathematical expression.
- Do not wrap it in calculator(...).
- Correct: 3100.0 / 1000
- Incorrect: calculator(3100.0 / 1000)

3. file_search
   Search filenames and text inside local workspace files.

Use this exact ReAct format.
For an action:

Thought: Briefly explain what you need.
Action: tool_name
Action Input: your input

For example:
Thought: I need the highest salinity measurement.
Action: sqlite_query
Action Input: SELECT stations.name, measurements.salinity_ppm 
              FROM measurements JOIN stations ON measurements.station_id = stations.id 
              ORDER BY measurements.salinity_ppm DESC LIMIT 1

The program will execute the action and return an Observation.
After receiving an Observation, continue reasoning.
You may use multiple tools in one task.

If a tool returns ERROR, understand the error and try
a corrected action when possible.

Do not invent database results or tool observations.

Always use the tools for questions that require actual
database queries, calculations, or file searches.

When you have enough information, respond:
Final Answer: your answer

Do not write your own Observation.
Do not use Markdown code fences for actions.
"""

# CONNECT TO OLLAMA
def ask_llm(messages):

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "stop": ["Observation:"]
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    return response.json()["message"]["content"]


# PARSE MODEL RESPONSE
def parse_reply(response):
    text = response.strip()
    text = text.replace("**", "")       # remove markdown bold

    final_match = re.search(
        r"(?im)^\s*Final Answer:\s*(.*)",
        text,
        re.DOTALL
    )

    action_match = re.search(
        r"(?im)^\s*Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*$",
        text
    )

    # Use whichever instruction appears first
    if final_match and (
        not action_match or final_match.start() < action_match.start()
    ):
        return "final", final_match.group(1).strip()

    if action_match:

        input_match = re.search(
            # parsing problem: send FinalAnswer & Action
            # r"(?im)^\s*Action Input:\s*(.*)",

            # stop at FinalAnswer - Thought - Action
            r"(?im)^\s*Action Input:\s*(.*?)(?=^\s*(?:Final Answer:|Thought:|Action:)|\Z)",
            text[action_match.end():],
            re.DOTALL
        )

        if not input_match:
            return "error", "Missing Action Input."

        tool_name = action_match.group(1).strip()
        tool_input = input_match.group(1).strip()

        if tool_input.startswith("```"):        # remove Code Fences
            tool_input = re.sub(
                r"^```[^\n]*\n?",
                "",
                tool_input
            )
            tool_input = re.sub(
                r"\n?```$",
                "",
                tool_input
            ).strip()

        if not tool_input:
            return "error", "Empty Action Input."

        return "action", (tool_name, tool_input)

    return "error", (
        "Invalid response format. "
        "Use Action / Action Input or Final Answer."
    )


# REACT AGENT LOOP
def run_agent(question, max_steps=MAX_STEPS, llm_function=None):

    if llm_function is None:            # so we can change the model response during test
        llm_function = ask_llm

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    executed_actions = set()
    successful_tools = 0

    print("Starting ReAct Agent")

    for step in range(1, max_steps + 1):
        print(f"\nStep {step}/{max_steps}")

        try:
            response = llm_function(messages)

            print("Agent response:")
            print(response)

            messages.append({
                "role": "assistant",
                "content": response
            })

            reply_type, result = parse_reply(response)

            if reply_type == "final":
                print("\nFinal Answer:")
                print(result)

                return result

            if reply_type == "error":
                observation = f"ERROR: {result}"
                print(f"\nObservation: {observation}")

                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
                continue

            tool_name, tool_input = result

            action_key = (
                tool_name,
                tool_input.strip().lower()
            )

            print(f"\nAction: {tool_name}")
            print(f"Action Input: {tool_input}")

            if action_key in executed_actions:
                observation = (
                    "ERROR: Repeated action detected. "
                    "Choose a different action or provide a final answer."
                )

            else:
                executed_actions.add(action_key)
                observation = run_tool(
                    tool_name,
                    tool_input
                )

                if not observation.startswith("ERROR:"):
                    successful_tools += 1

            print(f"Observation: {observation}")

            messages.append({
                "role": "user",
                "content": (
                    f"Observation: {observation[:1500]}\n"
                    "Continue the task using this result."
                )
            })

        except requests.RequestException as error:
            print(f"\nOllama connection error: {error}")
            return None

        except Exception as error:
            print(f"\nAgent error: {error}")
            return None

    print("\nSTOPPED: Maximum iteration limit reached.")
    print(f"Successful tool executions: {successful_tools}")
    return "STOPPED: Maximum iteration limit reached."


# SAFETY TESTS
def safety_checks():
    print("\nSAFETY CHECKS")
    tests = [
        (
            "Reject DROP TABLE",
            lambda: sqlite_query("DROP TABLE stations")
        ),
        (
            "Reject multiple SQL statements",
            lambda: sqlite_query(
                "SELECT * FROM stations; DELETE FROM stations"
            )
        ),
        (
            "Calculator",
            lambda: calculator("(45000 * 20) / 100")
        ),
        (
            "Reject arbitrary Python",
            lambda: calculator(
                "__import__('os').system('dir')"
            )
        ),
        (
            "Reject large exponent",
            lambda: calculator("2 ** 99999")
        ),
        (
            "Reject unknown tool",
            lambda: run_tool("weather_forecast", "Cairo")
        )
    ]

    for name, test_function in tests:
        print(f"\n{name}")

        try:
            result = test_function()
            print(result)

        except Exception as error:
            print(
                f"ERROR: {type(error).__name__}: {error}"
            )


# DETERMINISTIC ITERATION LIMIT TEST
def test_iteration_limit():
    print("\nITERATION LIMIT TEST")
    calls = []

    def fake_llm(messages):
        calls.append(1)

        return (
            "Thought: I will search another term.\n"
            "Action: file_search\n"
            f"Action Input: unique_probe_{len(calls)}"
        )

    result = run_agent(
        "Keep searching until the program stops.",
        max_steps=3,
        llm_function=fake_llm
    )

    assert len(calls) == 3, (
        "Iteration limit test failed."
    )

    assert result.startswith("STOPPED:"), (
        "Agent did not stop correctly."
    )

    print("\nPASS: Agent stopped after exactly 3 iterations.")


# REAL AGENT TESTS
# avoid hallucination in 5
TESTS = {

    "1": (
        "How many stations are in the database? "
        "Use sqlite_query."
    ),
    "2": (
        "Calculate the average of 120, 100, 150 and 90. "
        "Use calculator."
    ),

    "3": (
        "Search the workspace for the word water. "
        "Use file_search."
    ),
    "4": (
        "Find the highest salinity measurement in the database. "
        "Then calculate its value divided by 1000. "
        "You must use sqlite_query first and calculator second."
    ),

    "5": (
        "First try to use the invalid tool weather_forecast "
        "with input Cairo. After receiving the error, recover "
        "by querying the measurements table using the existing "
        "schema. Find the salinity for Aswan Gauge where "
        "month = '2026-03'. Do not use a year column."
    )
}


def run_tests(test_number="all"):

    if test_number == "6":
        test_iteration_limit()
        return

    if test_number == "all":
        selected_tests = list(TESTS.items())

    elif test_number in TESTS:
        selected_tests = [(
            test_number,
            TESTS[test_number]
        )]
    else:
        print("Unknown test number.")
        return

    for number, question in selected_tests:
        print(f"\n\nAGENT TEST {number}")
        run_agent(question)

# MAIN PROGRAM

if __name__ == "__main__":

    setup_environment()
    print("MILESTONE 5 - LOCAL REACT AGENT")

    print(f"Model: {MODEL}")
    print(f"Maximum iterations: {MAX_STEPS}")

    print("\nAvailable tools:")
    print("- sqlite_query")
    print("- calculator")
    print("- file_search")

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "safety":
            safety_checks()

        elif command == "test":
            test_number = (
                sys.argv[2].lower()
                if len(sys.argv) > 2
                else "all"
            )

            run_tests(test_number)
        else:
            question = " ".join(sys.argv[1:])
            run_agent(question)

    else:
        print("\nType 'exit' to quit.")

        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ("exit", "quit"):
                break

            if question:
                run_agent(question)
