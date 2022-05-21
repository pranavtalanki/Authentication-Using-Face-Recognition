from flask import Flask, Response, json, render_template, request
import cv2
import sqlite3
import os
import numpy as np
import face_recognition

app=Flask(__name__)
class usernames:
    name=""

newname=usernames()
known_face_names=[]
usernames1=[]
known_face_encodings=[]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
detector=cv2.CascadeClassifier("C:/Users/pranavtalanki/Desktop/academics/6th sem/AI project/proj2/haarcascade_frontalface_default.xml")
def addDetails(fname, lname, email, username, password, question, answer):
    con = sqlite3.connect("myData1.db")
    cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS Details(fname TEXT, lname TEXT, email TEXT, question TEXT, answer TEXT, username TEXT, password TEXT)')
    cur.execute("INSERT INTO Details VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (fname, lname, email, question, answer, username, password))
    con.commit()
    con.close()

def getDetails(username):
    con = sqlite3.connect("myData1.db")
    cur = con.cursor()
    username = str(username)
    result = cur.execute("SELECT question, answer FROM Details WHERE username == '%s'" % username)
    for ques, ans in result:
        print(ques, ans)
        return ques, ans
    return "No Question", "No"

@app.route('/storeData', methods=["GET"])
def store():
    fname = str(request.args.get("fname"))
    lname = str(request.args.get("lname"))
    email = str(request.args.get("password"))
    username = str(request.args.get("username"))
    password = str(request.args.get("password"))
    question = str(request.args.get("question"))
    answer = str(request.args.get("answer"))
    usernames1.append(username)
    #newname=usernames(username)
    newname.name=username
    addDetails(fname, lname, email, username, password, question, answer)
    return render_template('register_success.html', name=username)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames1(),mimetype='multipart/x-mixed-replace; boundary=frame')
def generate_frames1():
    
    global process_this_frame
    global known_face_encodings
    db_loc = "C:/Users/pranavtalanki/Desktop/academics/6th sem/AI project/proj2/images"
    directory = os.fsencode(db_loc)
    for file in os.listdir(directory):
            print("****")
            filename = os.fsdecode(file)
            print(filename,"+++++a+++++++++++++")
            known_face_names.append(filename.split('.')[0])
    for j in known_face_names:
        print (j,"........................")
        known_face_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(db_loc + "/" + i + ".png"))[0] for i in known_face_names]
    
    camera1=cv2.VideoCapture(0)
    while True:
        #reading frame
        success,frame=camera1.read()
        '''if not success:
            break
        else:'''

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
        if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                global face_names
                for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches: 
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if face_distances.all():
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
    
                    face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        ret,buff=cv2.imencode('.jpg',frame)
        frame=buff.tobytes()
        yield(b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
def generate_frames():
    camera=cv2.VideoCapture(0)
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            name=newname.name
            img_counter = 0
            name=name + ".png".format(img_counter)
            faces=detector.detectMultiScale(frame,1.1,7)
            #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                detected_face = cv2.resize(detected_face, (160, 160)) #resize to 224x224
                cv2.imwrite("C:/Users/pranavtalanki/Desktop/academics/6th sem/AI project/proj2/images/" + name, detected_face)
                print("{} written!".format(name))
            ret,buff=cv2.imencode('.jpg',frame)
            frame=buff.tobytes()
        yield(b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/registration_success')
def reg_success():
    return render_template('reg_success.html')

@app.route('/fail')
def fail():
    return render_template('reject.html')


@app.route('/logout')   
def reset_state():
    face_names = []
    return render_template('index.html')

@app.route('/welcome', methods=["GET"]) 
def welcome():
    name = request.args.get("userName")
    print(name)
    print(face_names)
    if name in face_names:
        ques, ans = getDetails(name)
        return render_template('welcome.html', name=name, question=ques, answer=ans)
    else:
        return render_template('reject.html')

@app.route('/')
def index():
        return render_template('index.html')
    
@app.route('/details')
def final_success():
    return render_template('details.html')

if __name__=="__main__":
    app.run()