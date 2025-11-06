from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import sqlite3, os, io, datetime
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import jsonify

lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

with open("intents.json") as f:
    intents = json.load(f)

app = Flask(__name__)
app.secret_key = "123"
DB_PATH = "database.db"

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    c = conn.cursor()
    c.execute("create table if not exists users(id integer primary key autoincrement,name text,email text unique,password text)")
    c.execute("""create table if not exists courses(
        id integer primary key autoincrement,
        title text,
        description text,
        video_url text,
        category text,
        order_index integer default 0
    )""")
    try:
        c.execute("ALTER TABLE courses ADD COLUMN order_index INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    c.execute("create table if not exists enrollments(id integer primary key autoincrement,user_id integer,course_id integer,completed integer default 0,unique(user_id,course_id))")
    c.execute("create table if not exists questions(id integer primary key autoincrement,course_id integer,question text,option1 text,option2 text,option3 text,option4 text,answer integer)")
    c.execute("create table if not exists marks(id integer primary key autoincrement,user_id integer,course_id integer,score integer,created_at text)")
    aptitude_courses = [
        ("Logical Aptitude", "Develop logical reasoning skills essential for aptitude tests", "/static/videos/logical.mp4", "Aptitude", 0),
        ("Quantitative Aptitude", "Master quantitative and mathematical problem-solving", "/static/videos/quantitative.mp4", "Aptitude", 1),
        ("Communication Aptitude", "Enhance verbal and communication abilities for assessments", "/static/videos/communication.mp4", "Aptitude", 2)
    ]

def current_user():
    if "user_id" in session:
        conn = db()
        u = conn.execute("select * from users where id=?", (session["user_id"],)).fetchone()
        conn.close()
        return u
    return None

def is_aptitude_completed(user_id):
    conn = db()
    apt_courses = conn.execute("select id from courses where category='Aptitude'").fetchall()
    if not apt_courses:
        conn.close()
        return True
    completed = True
    total_score = 0
    count = 0
    for ac in apt_courses:
        mark = conn.execute("select score from marks where user_id=? and course_id=? order by id desc limit 1", (user_id, ac['id'])).fetchone()
        if mark:
            total_score += mark['score']
            count += 1
        if not mark or mark['score'] < 10:
            completed = False
            break
    common_aptitude_score = total_score / max(count, 1) if count > 0 else 0
    conn.execute("CREATE TABLE IF NOT EXISTS aptitude_scores (user_id INTEGER PRIMARY KEY, common_score REAL)")
    conn.execute("INSERT OR REPLACE INTO aptitude_scores (user_id, common_score) VALUES (?, ?)", (user_id, common_aptitude_score))
    conn.commit()
    conn.close()
    return completed

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints):
    if len(ints) == 0:
        return "I am not sure how to answer that."
    tag = ints[0]["intent"]
    list_of_intents = intents["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            return np.random.choice(i["responses"])
    return "I am not sure how to answer that."

@app.route("/")
def index():
    return render_template("index.html", user=current_user())

@app.route("/student")
def student_index():
    u = current_user()
    if not u:
        return redirect(url_for("index"))
    conn = db()
    all_courses_query = conn.execute("select c.*, e.id as enrolled, e.completed from courses c left join enrollments e on e.course_id=c.id and e.user_id=? order by c.order_index ASC", (u["id"],)).fetchall()
    
    all_courses = [dict(row) for row in all_courses_query]
    
    last_marks = {}
    seen_courses = set()
    mark_rows = conn.execute("select course_id, score from marks where user_id=? order by created_at desc", (u["id"],)).fetchall()
    for row in mark_rows:
        cid = row['course_id']
        if cid not in seen_courses:
            last_marks[cid] = row['score']
            seen_courses.add(cid)
    
    for c in all_courses:
        c['last_score'] = last_marks.get(c['id'])
    
    aptitude_courses = [c for c in all_courses if c['category'] == 'Aptitude']
    non_aptitude_courses = [c for c in all_courses if c['category'] != 'Aptitude']

    aptitude_details = []
    all_apt_passed = True
    num_apt_enrolled = 0
    for c in aptitude_courses:
        enrolled = c['enrolled'] is not None
        video_done = enrolled and c['completed'] == 1
        score = c['last_score']
        quiz_done = score is not None
        passed = quiz_done and score >= 10
        if not passed:
            all_apt_passed = False
        status_class = 'completed' if passed else 'in-progress' if enrolled else 'locked'
        status_text = 'Completed' if passed else 'In Progress' if enrolled else 'Start Now'
        aptitude_details.append({
            'title': c['title'],
            'status_class': status_class,
            'status_text': status_text,
            'enrolled': enrolled,
            'course_id': c['id'],
            'score': score
        })
        if enrolled:
            num_apt_enrolled += 1
    
    aptitude_done = all_apt_passed or is_aptitude_completed(u['id'])  # Fallback to existing function
    aptitude_progress = (num_apt_enrolled / len(aptitude_courses) * 100) if aptitude_courses else 0
    
    conn.close()
    return render_template("student_index.html", user=u, aptitude_courses=aptitude_courses, non_aptitude_courses=non_aptitude_courses, aptitude_done=aptitude_done, aptitude_progress=aptitude_progress, aptitude_details=aptitude_details)

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name","").strip()
    email = request.form.get("email","").strip().lower()
    password = request.form.get("password","")
    if not name or not email or not password:
        flash("All fields required","danger")
        return redirect(url_for("index"))
    pw = generate_password_hash(password)
    try:
        conn = db()
        conn.execute("insert into users(name,email,password) values(?,?,?)",(name,email,pw))
        conn.commit()
        user = conn.execute("select * from users where email=?", (email,)).fetchone()
        conn.close()
        session["user_id"] = user["id"]
        return redirect(url_for("student_index"))
    except sqlite3.IntegrityError:
        flash("Email already registered","danger")
        return redirect(url_for("index"))

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email","").strip().lower()
    password = request.form.get("password","")
    conn = db()
    user = conn.execute("select * from users where email=?", (email,)).fetchone()
    conn.close()
    if user and check_password_hash(user["password"], password):
        session["user_id"] = user["id"]
        return redirect(url_for("student_index"))
    flash("Invalid credentials","danger")
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/admin_login", methods=["POST"])
def admin_login():
    username = request.form.get("username","").strip()
    password = request.form.get("password","").strip()
    if username == "admin" and password == "123":
        session["admin"] = True
        return redirect(url_for("admin_dashboard"))
    flash("Invalid admin credentials","danger")
    return redirect(url_for("index"))

@app.route("/admin_logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("index"))

@app.route("/admin")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("index"))
    conn = db()
    it = conn.execute("select * from courses where category='IT' order by order_index ASC").fetchall()
    biz = conn.execute("select * from courses where category='Business' order by order_index ASC").fetchall()
    apt = conn.execute("select * from courses where category='Aptitude' order by order_index ASC").fetchall()
    conn.close()
    return render_template("admin_dashboard.html", it_courses=it, biz_courses=biz, apt_courses=apt)

@app.route("/admin/course/new", methods=["POST"])
def admin_add_course():
    if not session.get("admin"):
        return redirect(url_for("index"))
    
    title = request.form.get("title","").strip()
    description = request.form.get("description","").strip()
    category = request.form.get("category","IT").strip()
    video_file = request.files.get("video_file")
    if not title or category not in ("IT","Business"):
        flash("Missing fields or invalid category (Aptitude courses are predefined and cannot be added).","danger")
        return redirect(url_for("admin_dashboard"))

    video_url = None
    if video_file and video_file.filename != "":
        filename = f"{datetime.datetime.utcnow().timestamp()}_{video_file.filename}"
        filepath = os.path.join("static", filename)
        video_file.save(filepath)
        video_url = "/" + filepath.replace("\\","/")
    conn = db()
    max_index = conn.execute("SELECT MAX(order_index) FROM courses WHERE category=?", (category,)).fetchone()[0] or 0
    next_index = max_index + 1

    conn.execute(
        "INSERT INTO courses(title,description,video_url,category,order_index) VALUES(?,?,?,?,?)",
        (title, description, video_url, category, next_index)
    )
    conn.commit()
    conn.close()
    flash("Course added","success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/course/update", methods=["POST"])
def admin_update_course():
    if not session.get("admin"):
        return redirect(url_for("index"))
    course_id = request.form.get("course_id")
    title = request.form.get("title","").strip()
    description = request.form.get("description","").strip()
    category = request.form.get("category","IT").strip()
    video_file = request.files.get("video_file")

    if not course_id or not title or category not in ("IT","Business","Aptitude"):
        flash("Missing required fields", "danger")
        return redirect(url_for("admin_dashboard"))

    conn = db()
    original = conn.execute("SELECT category FROM courses WHERE id=?", (course_id,)).fetchone()
    if not original:
        flash("Course not found", "danger")
        conn.close()
        return redirect(url_for("admin_dashboard"))
    
    original_category = original['category']
    if original_category != 'Aptitude' and category == 'Aptitude':
        apt_count = conn.execute("SELECT COUNT(*) FROM courses WHERE category='Aptitude'").fetchone()[0]
        if apt_count >= 3:
            flash("Cannot change category to Aptitude. Maximum of 3 Aptitude courses allowed (Logical, Quantitative, and Communication are predefined).", "danger")
            conn.close()
            return redirect(url_for("admin_dashboard"))

    video_url = None
    if video_file and video_file.filename != "":
        filename = f"{datetime.datetime.utcnow().timestamp()}_{video_file.filename}"
        filepath = os.path.join("static", filename)
        video_file.save(filepath)
        video_url = "/" + filepath.replace("\\","/")

    
    if video_url:
        conn.execute("UPDATE courses SET title=?, description=?, category=?, video_url=? WHERE id=?",
                     (title, description, category, video_url, course_id))
    else:
        conn.execute("UPDATE courses SET title=?, description=?, category=? WHERE id=?",
                     (title, description, category, course_id))
    conn.commit()
    conn.close()
    flash("Course updated successfully", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/course/<int:course_id>/delete", methods=["POST"])
def admin_delete_course(course_id):
    if not session.get("admin"):
        return redirect(url_for("index"))
    conn = db()
    conn.execute("DELETE FROM courses WHERE id=?", (course_id,))
    conn.execute("DELETE FROM questions WHERE course_id=?", (course_id,))
    conn.execute("DELETE FROM enrollments WHERE course_id=?", (course_id,))
    conn.commit()
    conn.close()
    flash("Course deleted", "success")
    return redirect(url_for("admin_dashboard"))

@app.route('/admin/save_course_order', methods=['POST'])
def admin_save_course_order():
    data = request.get_json()
    course_ids = data.get('course_ids', [])
    category = data.get('category')

    conn = db()
    for idx, cid in enumerate(course_ids):
        conn.execute("UPDATE courses SET order_index=? WHERE id=? AND category=?", (idx, cid, category))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Order updated"})

@app.route("/admin/clean_duplicates", methods=["GET", "POST"])
def clean_duplicates():
    if not session.get("admin"):
        flash("Admin access required", "danger")
        return redirect(url_for("admin_dashboard"))
    
    if request.method == "POST": 
        conn = db()
        c = conn.cursor()
        
        
        aptitude_titles = {
            "Logical Aptitude": None,
            "Quantitative Aptitude": None,
            "Communication Aptitude": None
        }
        
       
        apt_courses = c.execute("SELECT id, title FROM courses WHERE category='Aptitude' ORDER BY id ASC").fetchall()
        
        to_keep = set()
        for course in apt_courses:
            title = course['title']
            if title in aptitude_titles and aptitude_titles[title] is None:
                aptitude_titles[title] = course['id']
                to_keep.add(course['id'])
        
        to_delete = [course['id'] for course in apt_courses if course['id'] not in to_keep]
        for cid in to_delete:
            c.execute("DELETE FROM questions WHERE course_id=?", (cid,))
            c.execute("DELETE FROM enrollments WHERE course_id=?", (cid,))
            c.execute("DELETE FROM marks WHERE course_id=?", (cid,))
            c.execute("DELETE FROM courses WHERE id=?", (cid,))
        
       
        for category in ["IT", "Business"]:
            cat_courses = c.execute("SELECT id FROM courses WHERE category=? ORDER BY id ASC", (category,)).fetchall()
            if len(cat_courses) > 3:
                extras = [course['id'] for course in cat_courses[3:]]  
                for cid in extras:
                    c.execute("DELETE FROM questions WHERE course_id=?", (cid,))
                    c.execute("DELETE FROM enrollments WHERE course_id=?", (cid,))
                    c.execute("DELETE FROM marks WHERE course_id=?", (cid,))
                    c.execute("DELETE FROM courses WHERE id=?", (cid,))
        
        for category in ["IT", "Business", "Aptitude"]:
            cat_courses = c.execute("SELECT id FROM courses WHERE category=? ORDER BY id ASC LIMIT 3", (category,)).fetchall()
            for idx, course in enumerate(cat_courses):
                c.execute("UPDATE courses SET order_index=? WHERE id=?", (idx, course['id']))
        
        conn.commit()
        conn.close()
        conn = db()
        counts = {
            "IT": conn.execute("SELECT COUNT(*) FROM courses WHERE category='IT'").fetchone()[0],
            "Business": conn.execute("SELECT COUNT(*) FROM courses WHERE category='Business'").fetchone()[0],
            "Aptitude": conn.execute("SELECT COUNT(*) FROM courses WHERE category='Aptitude'").fetchone()[0]
        }
        conn.close()
        
        flash(f"Cleaning complete! Now balanced: IT={counts['IT']}, Business={counts['Business']}, Aptitude={counts['Aptitude']} (duplicates removed).", "success")
        return redirect(url_for("admin_dashboard"))
    
    return render_template("confirm_clean.html", message="This will remove all duplicate Aptitude courses")


@app.route("/admin/course/<int:course_id>/questions")
def admin_questions(course_id):
    if not session.get("admin"):
        return redirect(url_for("index"))
    conn = db()
    course = conn.execute("select * from courses where id=?", (course_id,)).fetchone()
    qs = conn.execute("select * from questions where course_id=? order by id", (course_id,)).fetchall()
    conn.close()
    if not course:
        return redirect(url_for("admin_dashboard"))
    return render_template("admin_questions.html", course=course, questions=qs)

@app.route("/admin/course/<int:course_id>/questions", methods=["POST"])
def admin_add_question(course_id):
    if not session.get("admin"):
        return redirect(url_for("index"))
    q = request.form.get("question","").strip()
    o1 = request.form.get("option1","").strip()
    o2 = request.form.get("option2","").strip()
    o3 = request.form.get("option3","").strip()
    o4 = request.form.get("option4","").strip()
    ans = int(request.form.get("answer","1"))
    if not q or not o1 or not o2 or not o3 or not o4 or ans not in (1,2,3,4):
        flash("Invalid question","danger")
        return redirect(url_for("admin_questions", course_id=course_id))
    conn = db()
    conn.execute("insert into questions(course_id,question,option1,option2,option3,option4,answer) values(?,?,?,?,?,?,?)",(course_id,q,o1,o2,o3,o4,ans))
    conn.commit()
    conn.close()
    flash("Question added","success")
    return redirect(url_for("admin_questions", course_id=course_id))

@app.route("/admin/question/<int:question_id>/edit", methods=["POST", "GET"])
def admin_edit_question(question_id):
    if not session.get("admin"):
        return redirect(url_for("index"))
    conn = db()
    q = conn.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        option1 = request.form.get("option1", "").strip()
        option2 = request.form.get("option2", "").strip()
        option3 = request.form.get("option3", "").strip()
        option4 = request.form.get("option4", "").strip()
        answer = int(request.form.get("answer", 1))
        conn.execute("""UPDATE questions
                        SET question=?, option1=?, option2=?, option3=?, option4=?, answer=?
                        WHERE id=?""",
                     (question, option1, option2, option3, option4, answer, question_id))
        conn.commit()
        conn.close()
        flash("Question updated", "success")
        return redirect(url_for("admin_questions", course_id=q["course_id"]))
    conn.close()
    return render_template("admin_edit_question.html", question=q)

@app.route("/admin/question/<int:question_id>/delete", methods=["POST"])
def admin_delete_question(question_id):
    if not session.get("admin"):
        return redirect(url_for("index"))
    
    conn = db()
    q = conn.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
    if q:
        conn.execute("DELETE FROM questions WHERE id=?", (question_id,))
        conn.commit()
        flash("Question deleted", "success")
        course_id = q["course_id"]
    else:
        flash("Question not found", "danger")
        course_id = 0
    conn.close()
    return redirect(url_for("admin_questions", course_id=course_id))


@app.route("/courses/enroll/<int:course_id>", methods=["POST"])
def enroll(course_id):
    u = current_user()
    if not u:
        return redirect(url_for("index"))
    conn = db()
    course = conn.execute("select category from courses where id=?", (course_id,)).fetchone()
    conn.close()
    if course['category'] != 'Aptitude' and not is_aptitude_completed(u['id']):
        flash("Complete all Aptitude modules first to enroll in other courses.", "danger")
        return redirect(url_for("student_index"))
    conn = db()
    try:
        conn.execute("insert into enrollments(user_id,course_id,completed) values(?,?,0)", (u["id"],course_id))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()
    return redirect(url_for("course_detail", course_id=course_id))

@app.route("/courses/<int:course_id>")
def course_detail(course_id):
    u = current_user()
    if not u:
        return redirect(url_for("index"))
    conn = db()
    c = conn.execute("select * from courses where id=?", (course_id,)).fetchone()
    e = conn.execute("select * from enrollments where user_id=? and course_id=?", (u["id"], course_id)).fetchone()
    conn.close()
    if not c:
        return redirect(url_for("student_index"))
    if c['category'] != 'Aptitude' and not is_aptitude_completed(u['id']):
        flash("Complete all Aptitude modules first.", "danger")
        return redirect(url_for("student_index"))
    return render_template("course_detail.html", user=u, course=c, enrollment=e)

@app.route("/courses/<int:course_id>/complete_video", methods=["POST"])
def complete_video(course_id):
    u = current_user()
    if not u:
        return redirect(url_for("index"))
    conn = db()
    course = conn.execute("select category from courses where id=?", (course_id,)).fetchone()
    conn.close()
    if course['category'] != 'Aptitude' and not is_aptitude_completed(u['id']):
        flash("Complete all Aptitude modules first.", "danger")
        return redirect(url_for("student_index"))
    conn = db()
    conn.execute("update enrollments set completed=1 where user_id=? and course_id=?", (u["id"],course_id))
    conn.commit()
    conn.close()
    return redirect(url_for("quiz", course_id=course_id))

@app.route("/quiz/<int:course_id>")
def quiz(course_id):
    u = current_user()
    if not u:
        return redirect(url_for("index"))
    conn = db()
    e = conn.execute("select * from enrollments where user_id=? and course_id=?", (u["id"],course_id)).fetchone()
    if not e:
        conn.close()
        return redirect(url_for("student_index"))
    qs = conn.execute("select * from questions where course_id=? order by id", (course_id,)).fetchall()
    course = conn.execute("select * from courses where id=?", (course_id,)).fetchone()
    last_mark = conn.execute("select * from marks where user_id=? and course_id=? order by id desc limit 1",(u["id"],course_id)).fetchone()
    conn.close()
    return render_template("quiz.html", user=u, course=course, questions=qs, last_mark=last_mark)

@app.route("/quiz/<int:course_id>/submit", methods=["POST"])
def submit_quiz(course_id):
    u = current_user()
    if not u:
        return redirect(url_for("index"))
    conn = db()
    qs = conn.execute("select id,answer from questions where course_id=? order by id", (course_id,)).fetchall()
    score = 0
    for q in qs:
        key = f"q_{q['id']}"
        val = request.form.get(key)
        if val and int(val) == q["answer"]:
            score += 1
    conn.execute("insert into marks(user_id,course_id,score,created_at) values(?,?,?,?)",(u["id"],course_id,score,datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return redirect(url_for("certificate", course_id=course_id))

@app.route("/certificate/<int:course_id>")
def certificate(course_id):
    u = current_user()
    if not u:
        return redirect(url_for("index"))
    conn = db()
    course = conn.execute("select * from courses where id=?", (course_id,)).fetchone()
    mark = conn.execute("select * from marks where user_id=? and course_id=? order by id desc limit 1",(u["id"],course_id)).fetchone()
    conn.close()
    if not course or not mark:
        return redirect(url_for("student_index"))
    eligible = mark["score"] >= 10
    return render_template("certificate.html", user=u, course=course, mark=mark, eligible=eligible)

from reportlab.lib.colors import HexColor
from reportlab.lib.units import cm

@app.route("/certificate/<int:course_id>/download")
def download_certificate(course_id):
    u = current_user()
    if not u:
        return redirect(url_for("index"))

    conn = db()
    course = conn.execute("SELECT * FROM courses WHERE id=?", (course_id,)).fetchone()
    mark = conn.execute(
        "SELECT * FROM marks WHERE user_id=? AND course_id=? ORDER BY id DESC LIMIT 1",
        (u["id"], course_id)
    ).fetchone()
    conn.close()

    if not course or not mark or mark["score"] < 10:
        return redirect(url_for("certificate", course_id=course_id))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4


    c.setFillColor(HexColor("#fffdf6"))
    c.rect(0, 0, w, h, fill=1, stroke=0)

    
    c.setFont("Helvetica-Bold", 80)
    c.setFillColor(HexColor("#f0e6d6"))
    for y in range(100, int(h), 200):
        for x in range(100, int(w), 200):
            c.drawCentredString(x, y, "★")

    
    margin = 2*cm
    c.setStrokeColor(HexColor("#d4af37"))
    c.setLineWidth(4)
    c.rect(margin, margin, w-2*margin, h-2*margin, stroke=1, fill=0)

    
    for i, color in enumerate(["#ffd700", "#ffc200", "#ffae00"]):
        c.setFillColor(HexColor(color))
        c.roundRect(w/2 - 130, h-150+i*5, 260, 15, 7, fill=1, stroke=0)

    
    c.setFont("Helvetica-Bold", 28)
    c.setFillColor(HexColor("#b8860b"))
    c.drawCentredString(w/2 + 2, h-142, "Certificate of Completion")
    c.setFillColor(HexColor("#ffffff"))
    c.drawCentredString(w/2, h-140, "Certificate of Completion")

    
    c.setFont("Helvetica", 14)
    c.setFillColor(HexColor("#333333"))
    c.drawCentredString(w/2, h-180, "This certifies that")

    c.setFont("Helvetica-Bold", 26)
    c.setFillColor(HexColor("#b8860b"))
    c.drawCentredString(w/2, h-220, u["name"])

    
    c.setFont("Helvetica", 16)
    c.setFillColor(HexColor("#333333"))
    c.drawCentredString(w/2, h-255, "has successfully completed the course")

    c.setFont("Helvetica-BoldOblique", 20)
    c.setFillColor(HexColor("#8b0000"))
    c.drawCentredString(w/2, h-285, course["title"])


    c.setFont("Helvetica", 12)
    c.setFillColor(HexColor("#555555"))
    c.drawCentredString(w/2, h-320, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")


    c.setLineWidth(1.5)
    c.line(80, 120, w/2-40, 120)
    c.line(w/2+40, 120, w-80, 120)
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString((80 + w/2-40)/2, 110, "Instructor")
    c.drawCentredString((w/2+40 + w-80)/2, 110, "Authorized Signature")

    
    c.setStrokeColor(HexColor("#d4af37"))
    c.setLineWidth(2)
    seal_x, seal_y = w-80, h-180
    c.circle(seal_x, seal_y, 40, stroke=1, fill=0)
    for angle in range(0, 360, 36):
        rad = angle * 3.1416 / 180
        x2 = seal_x + 30 * cm * 0.01 * 3 * (0.5**0.5)
        y2 = seal_y
        c.line(seal_x, seal_y, x2, y2)

    c.showPage()
    c.save()
    buf.seek(0)

    return send_file(buf, as_attachment=True, download_name=f"certificate_{course_id}.pdf", mimetype="application/pdf")

@app.route("/chatbot", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    message = data.get("message", "")
    ints = predict_class(message)
    res = get_response(ints)
    return jsonify({"response": res})

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    app.run(debug=False, port=900)
