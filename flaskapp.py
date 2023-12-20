from admissionpredictionlib import prediction
from flask import Flask, request, render_template

app = Flask(__name__)
@app.route("/", methods = ["GET", "POST"])
def gfg():
	name = request.form.get("name")
	gre = int(request.form.get("gre"))
	toefl = request.form.get("toefl")
	rating = request.form.get("rating")
	sop = request.form.get("sop")
	lor = request.form.get("lor")
	gpa = request.form.get("gpa")
	research = request.form.get("research")
	result = prediction(gre,toefl,rating,sop,lor,gpa,research)
	return "Hello!"+ name+"your chances of getting the admission are"+str(result)

if __name__ == "__main__":
	app.run(debug=True)