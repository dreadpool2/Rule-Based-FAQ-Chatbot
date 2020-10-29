from flask import Flask, render_template, request
import ChatbotManager as cm 


app = Flask(__name__)
dialogue_manager = cm.ChatbotManager()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/agents/")
def agents():
    return render_template("agents.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(dialogue_manager.generate_faq_answer(userText))

@app.route("/getHelp")
def send_agent_help():
    user = request.args.get('user')
    question = request.args.get('question')

    return str(dialogue_manager.get_agent_help(user, question))

##In agents page
@app.route("/agentHelp")
def get_live_queries():
    return str(dialogue_manager.get_all_data())

##index page when question and answer received from agent
@app.route("/receiveHelp")
def get_help():
    userText = request.args.get('user')
    return dialogue_manager.check_replies(userText)

@app.route("/agentSentHelp")
def get_real_agent_help():
    userName = request.args.get('userName')
    answer = request.args.get('answer')
    status = request.args.get('status')

    return str(dialogue_manager.update_data_agent(userName, answer, status))

@app.route("/dropAll")
def drop_all():
    dialogue_manager.drop_all_queries()
    
@app.route("/adminPortal/")
def adminPortal():
    return render_template("adminPortal.html")

@app.route("/addData")
def add_data():
    ques = request.args.get('question')
    ans = request.args.get('answer')
    
    dialogue_manager.add_data(ques, ans)

#if __name__ == "__main__":
#    app.run()