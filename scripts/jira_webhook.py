from flask import Flask, request, jsonify
import pandas as pd
import datetime
import os

app = Flask(__name__)
WEBHOOK_LOG_PATH = "../data/webhook_tickets.csv"

# Ensure file exists with headers
if not os.path.exists(WEBHOOK_LOG_PATH):
    pd.DataFrame(columns=[
        "Ticket_ID", "Customer_Description", "Product_Module",
        "Error_Nemonics", "Tags", "Timestamp"
    ]).to_csv(WEBHOOK_LOG_PATH, index=False)

@app.route("/jira_webhook", methods=["POST"])
def jira_webhook():
    try:
        data = request.json
        ticket_id = data.get("ticket_id")
        description = data.get("description")
        product_module = data.get("product_module", "Unknown")
        error = data.get("error_nemonics", "")
        tags = ",".join(data.get("tags", []))

        timestamp = datetime.datetime.now().isoformat()

        df = pd.read_csv(WEBHOOK_LOG_PATH)
        df = pd.concat([
            df,
            pd.DataFrame([{
                "Ticket_ID": ticket_id,
                "Customer_Description": description,
                "Product_Module": product_module,
                "Error_Nemonics": error,
                "Tags": tags,
                "Timestamp": timestamp
            }])
        ])
        df.to_csv(WEBHOOK_LOG_PATH, index=False)

        return jsonify({"status": "received", "ticket_id": ticket_id})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
