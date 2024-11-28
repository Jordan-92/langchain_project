from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from tools import tools
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "http://127.0.0.1:5500"}})

# Initialize LangChain
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.route("/ask", methods=["GET"])
def ask_get():
    question = request.args.get("question")
    if not question:
        return jsonify({"error": "Query parameter 'question' is required."}), 400
    try:
        response = agent_executor.invoke({"input": question})
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
